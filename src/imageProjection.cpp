#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "jsk_recognition_msgs/BoundingBox.h"
#include "jsk_recognition_msgs/BoundingBoxArray.h"
#include "mymsgs/PointWithBBox.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subBBox;            //订阅检测框BoundingBoxArray
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    jsk_recognition_msgs::BoundingBoxArray currentBBboxMsg;
    std::deque<jsk_recognition_msgs::BoundingBoxArray> BBboxQueue;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;


public:
    ImageProjection():
    deskewFlag(0)
    {
        // 订阅imu数据，里程计数据，原始点云数据
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        
        // 订阅点云和检测框二合一数据
        subLaserCloud = nh.subscribe<mymsgs::PointWithBBox>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());


        //subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布运动补偿后的点云，这个主要用来rviz可视化
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        // 发布运动补偿后的点云和检测框二合一信息，这个被FeatureExtration订阅。
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    /**
     * @description: 分配内存，避免野指针
     * @return {*}
     */
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}

    /**
     * @description: imu回调函数
     * @param {ConstPtr&} imuMsg
     * @return {*}
     */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        // imu数据坐标转换
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        //线程加锁，把imu数据保存进入队列
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    /**
     * @description: odom回调，直接将odometryMsg加入到odomQueue队列中
     * @param {ConstPtr&} odometryMsg
     * @return {*}
     */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    /**
     * @description: 点云+检测框PointWithBBox消息的回调
     * @param {PointCloud2ConstPtr&} laserCloudMsg 原始的激光雷达的点云话题信息
     * @return {*}
     */
    void cloudHandler(const mymsgs::PointWithBBoxConstPtr& PointWithBBox)
    {
        // 先处理里面的点云
        // 取出消息里面的点云消息
        const sensor_msgs::PointCloud2 laserCloudMsg = PointWithBBox->CloudMsg;
        //取出检测框数组
        const jsk_recognition_msgs::BoundingBoxArray bboxarray = PointWithBBox->BBboxArray;
        
        // 缓存点云消息，并且检查是否有两帧以上的点云，检查点云格式是否合格，否则在这里就直接返回了
        if (!cachePointCloudWithBBbox(laserCloudMsg, bboxarray))
            return;

        //获取运动补偿所需要的信息
        if (!deskewInfo())
            return;

        projectPointCloud();//将点云投影到一个矩形上，并且保存每个点的信息

        cloudExtraction();//提取有效的点云信息

        publishClouds();//将点云发布出去

        resetParameters();//参数复位
    }

    /**
     * @description: 将点云数据放入队列缓存，并且对点云帧的消息格式进行检查
     * @param {PointCloud2ConstPtr&} laserCloudMsg
     * @return {*}
     */
    bool cachePointCloudWithBBbox(const sensor_msgs::PointCloud2 laserCloudMsg, jsk_recognition_msgs::BoundingBoxArray bboxarray)
    {

        // 检测框入队列
        BBboxQueue.push_back(bboxarray);

        // cache point cloud
        cloudQueue.push_back(laserCloudMsg);
        // 确保队列里有大于2帧以上的点云数量
        if (cloudQueue.size() <= 2)
            return false;

        // 缓存了足够多的点云
        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());//拿出来第一帧
        currentBBboxMsg = std::move(BBboxQueue.front());//拿出来第一帧

        cloudQueue.pop_front();// 从队列中删除
        BBboxQueue.pop_front();


        // 如果是VELODYNE的雷达，直接使用pcl的库将ros格式转换成pcl的点云格式
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // 如果是OUSTER的激光雷达
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            // 不支持其他类型的雷达
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();// 最早的一个点的时间（header里面是最早的一个点的时间）
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; //最晚的一个点的绝对时间。laserCloudIn->points.back().time是相对于该帧中的第一个点的相对时间

        // check dense flag
        // is_dense标志位，表示该帧点云是否是有序排列的
        if (laserCloudIn->is_dense == false)
        {
            // 要求必须是有序排列的点云
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        // 查看驱动里是否有每个点属哪一根扫描线scan这个信息
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                // 如果没有这个信息就需要向loam或者lego loam那样手动计算scan id， 现在velodyne的驱动里都会带这些信息
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        // 同样 检查是否有时间戳信息
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /**
     * @description: 获取运动补偿所需要的信息
     * @return {*}
     */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 确保imu数据覆盖这一帧的点云
        // imu队列不为空，imu队列中最早的比点云中最早的点还早，最晚的imu比点云中最晚的点还晚
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        //使用imu和odom都可以对点云进行补偿，imu只能补偿旋转，odom除了能提供点云帧的初始位姿，还可以在平移旋转两方面进行补偿
        imuDeskewInfo();//准备利用imu补偿点云帧的信息
        odomDeskewInfo();//准备利用odom补偿点云帧的信息

        return true;
    }

    /**
     * @description: 准备imu补偿的信息
     * @return {*}
     */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        //扔掉太老的imu信息。老imu信息没有意义，同时避免队列过大
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01) //这里有个10ms的余量。
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        // 遍历imu队列中的所有imu信息，计算imuRotX[],imuRotY[],imuRotZ[]
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            //如果这个imu数据的时间在点云第一个点的时间之前
            if (currentImuTime <= timeScanCur)
            // 取出来thisImuMsg中的角度，赋给三个变量
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)//到了覆盖该帧的最后一个imu数据了，跳出
                break;

            if (imuPointerCur == 0){
                //如果是第一个imu数据，就全部设置为0
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            //如果不是第一个imu数据了，就可以进行一个简单的积分
            // get angular velocity
            double angular_x, angular_y, angular_z;
            // 取出来thisImuMsg中的角速度，赋给三个变量
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;//循环里面多加了一个（break的时候已经是不需要的imu数据了），减1

        if (imuPointerCur <= 0)
            return;

        // 可以使用imu数据进行数据补偿
        cloudInfo.imuAvailable = true;
    }

    /**
     * @description: 使用odom数据，提供点云的起始位姿和用来做运动补偿
     * @return {*}
     */
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            // 扔掉过早的数据
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        // 点云时间        front ********  end
        // odom时间        front   *******  end
        // 最早的odom必须是比整个点云的第一个点还要早，否则肯定是不能覆盖整个点云时间的，那就不能用来提供点云的起始位姿
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        //遍历队列，找到对应该帧点云中最早点的时间的odom数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        //将ros话题消息的姿态转换成tf的格式
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);
        //然后将四元数转换为欧拉角
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        // 记录点云起始时刻对应的odom姿态
        // Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;// odom提供了这一帧点云的初始位姿

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        //如果要用odom数据来做运动补偿，那么就odom队列就必须覆盖点云帧的最后一个点，
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        // 找到点云最晚时间对应的odom数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        // 如果odom退化了，置信度不高，就不能用来做运动补偿了
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 起始位姿和结束位姿都转换成Affine3f数据格式
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 计算起始位姿和结束位姿之间的delta pose
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 将这个增量转换成xyz和欧拉角的形式
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;// 表示可以用odom来做运动补偿
    }

    /**
     * @description: 获取相对时刻为pointTime的lidar点相对于该帧点云起始点的旋转
     * @param {double} pointTime
     * @param {float} *rotXCur
     * @param {float} *rotYCur
     * @param {float} *rotZCur
     * @return {*}
     */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        // imuPointerCur在之前已经被赋值为了覆盖该点云帧的最早一个imu数据的序号
        // imuPointerCur是imu计算的旋转buffer的总共大小，这里用的就是一种朴素的确保不越界的方法
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // 如果时间戳不在两个imu的旋转之间，就直接赋值了
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 否则，做一个线性插值，得到相对旋转
            // imuPointerBack   imuPointerFront
            //      *                  *
            //               *
            //         imuPointerCur

            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    /**
     * @description: 平移运动补偿，这里其实没做。因为作者的注释里面认为没什么必要
     * @param {double} relTime
     * @param {float} *posXCur
     * @param {float} *posYCur
     * @param {float} *posZCur
     * @return {*}
     */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    /**
     * @description: 运动补偿
     * @param {PointType} *point
     * @param {double} relTime 该点相对于点云帧起始时刻的相对时间
     * @return {*}
     */
    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;//算该点的绝对时间

        float rotXCur, rotYCur, rotZCur;
        // 计算当前点相对于起始点的相对旋转
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        // 平移运动补偿，其实是没做
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            //计算第一个点的相对位姿
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        //计算当前点和第一个点的相对位姿
        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        // 就是R * p + t 把点补偿到第一个点对应时刻时的位姿
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    /**
     * @description: 将点云投影到一个矩阵上，并且保存每个点的信息
     * @return {*}
     */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        // 遍历整个点云帧
        for (int i = 0; i < cloudSize; ++i)
        {
            //取出对应的某个点
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            // 计算这个点离lidar中心的距离
            float range = pointDistance(thisPoint);
            //离lidar中心太近或者太远都认为是异常值，跳过
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            //去除对应的在第几根scan上面
            int rowIdn = laserCloudIn->points[i].ring;
            // scan id必须合理
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            //如果雷达线数太多，需要降采样，那么就根据scan id适当跳过
            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                // 计算水平角
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                // 计算水平线束id，转换到x负方向e为起始，顺时针为正方向，范围[0,H]
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }
            //对水平id进行检查
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            //如果这个位置已经有了填充就跳过
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            //对点做运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            //将点的距离数据信息保存到range矩阵中
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 算出这个点的索引
            int index = columnIdn + rowIdn * Horizon_SCAN;
            // 保存这个点的坐标
            fullCloud->points[index] = thisPoint;
        }
    }

    // 提取出有效点的信息
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 遍历每一根扫描线scan
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 算一下这条scan可以计算曲率的起始点（计算曲率需要左右各五个点）
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    //如果这是一个有用的点
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;//这个点对应哪一根垂直线
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);// 距离信息
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);//3d坐标信息
                    // size of extracted cloud
                    ++count;
                }
            }
            //这条scan可以计算曲率的终点
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    /**
     * @description: 将点云发布出去，其实是发给featureExtraction.cpp的
     * @return {*}
     */    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.BBoxArray = currentBBboxMsg;
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
