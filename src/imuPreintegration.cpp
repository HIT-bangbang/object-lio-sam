#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    // 如果lidar帧和baselink帧不是同一个坐标系
    TransformFusion()
    {
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                // 查询一下lidar和baselink的TF坐标变化
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }

        // 订阅地图优化节点的全局位姿和预积分节点的增量位姿
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);
    }

    /**
     * @description: 把ros的消息格式转换为eigen的格式
     * @param {Odometry} odom
     * @return {*}
     */    
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);// 通过PCL库进行转化，PCL里面用的其实也是eigen的数据结构
    }

    
    /**
     * @description: 全局位姿（带回环）保存下来
     * @param {ConstPtr&} odomMsg
     * @return {*}
     */
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = odomMsg->header.stamp.toSec();//记录时间
    }

    /**
     * @description: subImuOdometry的回调函数，订阅imu预积分节点发布的位姿，将其补偿到
     * @param {ConstPtr&} odomMsg imu预计分节点发布的位姿。 imuHandler中发布的imu数据
     * @return {*}
     */
    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // static tf
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));// 建图过程中认为map和odom是重合的
        // 发送静态tf，odom和map系将他们重合
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        // imu得到的里程计结果入队列
        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        // 如果没有受到lidar位姿就return
        if (lidarOdomTime == -1)
            return;
        // 否则，受到了位姿，弹出最新lidar位姿时刻之前的imu里程计数据（最新lidar位姿之前的老数据没用了全部扔掉）
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        // 计算队列里imu积分里程计的增量
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        // 将这个增量补偿到lidar的位姿上去，就得到了最新的预测的位姿
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        // 分解成平移+欧拉角的方式
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        // 发送全局一致的最新位姿
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        // 更新tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);// 将话题消息转换为TF
        //如果lidar和baselink坐标系不同，还需要有个位姿变换
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
            // 更新odom到baselink的tf
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // 发送imu里程计的轨迹(rviz中蓝色轨迹前面的一小段紫色的轨迹)
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        //控制一下发布频率，不要超过10hz
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            //将最新的位姿送入到轨迹类型中
            imuPath.poses.push_back(pose_stamped);
            // 把lidar帧之前的轨迹全部擦掉。这个轨迹是用imu预测的轨迹，起点是最新的lidar关键帧，终点是最新的imu帧。
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            // 发布轨迹，这个轨迹实际上是可视化imu预计分节点输出的预测值
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;
    
    // T_bl: tramsform points from lidar frame to imu frame 
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    // T_lb: tramsform points from imu frame to lidar frame
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration()
    {
        //订阅imu话题
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅后端优化的节点发布的消息，进入里程计的回调
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);//使用东北天（ENU）坐标系，重力方向朝下 -g
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        //速度积分得到位置的噪声
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        //初始位姿置信度设置比较高
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        //初始速度执行度就设置的差一些
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        //零偏的执行度也设置的高一些
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        // 正常情况下，lidar odom的协方差矩阵
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        //lidar odom退化之后的协方差矩阵
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        // 两帧bias的协方差矩阵
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        // 这里new了两个预积分类也是一个用来预积分优化，一个用来推算最新位姿的
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    /**
     * @description: 重置整个优化
     * @return {*}
     */
    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    /**
     * @description: 重置参数
     * @return {*}
     */
    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);// ROS_TIME取出odomMsg消息的时间戳

        // make sure we have imu data to integrate
        // 确保imu队列中有数据
        if (imuQueOpt.empty())
            return;
        //获取里程计位姿
        //平移
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        //旋转
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        //该位姿是否出现退化，一旦pose.covariance[0]置为1就代表有退化的风险
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        //把位姿转换成gtsam格式
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        // 首先初始化系统
        if (systemInitialized == false)
        {
            //将优化问题复位
            resetOptimization();

            // pop old IMU message
            // 将这个里程计之前的imu信息全部扔掉
            while (!imuQueOpt.empty())
            {
                //从队列的最前面（最老的imu数据）开始，如果比当前时间小一点，就扔掉
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            // 将lidar的位姿转换到imu坐标系下，因为所有的操作都是在imu坐标系下进行的
            prevPose_ = lidarPose.compose(lidar2Imu);
            //设置其初始位姿和置信度
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);//对第0个位姿X(0)的先验约束，其先验值为prevPose_，置信度（协方差）为priorPoseNoise
            // 将约束加入到因子中
            graphFactors.add(priorPose);
            // initial velocity
            //初始化速度，直接赋0了
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);//这里速度的协方差设置的比较大，也就是置信度比较小
            //将对速度的约束也加入到因子图中
            graphFactors.add(priorVel);
            // initial bias
            //初始化第0个零偏B(0),设置先验值为prevBias_，协方差为priorBiasNoise
            prevBias_ = gtsam::imuBias::ConstantBias();//全部为0
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            //将零偏的约束也加入到因子图中
            graphFactors.add(priorBias);

            //以上添加约束完成，下面添加状态量

            // add values
            //将各个状态量赋值为初始值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            // 约束和状态量更新送进isam优化器
            optimizer.update(graphFactors, graphValues);
            // 送进优化器之后，保存约束和状态量的变量清零
            graphFactors.resize(0);
            graphValues.clear();
            
            //预积分的接口，使用初始零偏进行初始化
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;//将标志位设置为true，后面就不会再进行初始化了
            return;
        }


        // reset graph for speed
        //当isam优化器中加入了较多的约束后，为了避免运算时间变长，就直接清空，冲重新开始
        if (key == 100)
        {
            // get updated noise before reset
            // 取出上一时刻的协方差矩阵（速度位姿零偏）
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            //复位整个优化问题
            resetOptimization();
            // 将最新时刻的位姿速度零偏以及对应的协方差矩阵加入到因子图中
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        // 将两帧之间的imu积分
        while (!imuQueOpt.empty())
        {
            // 如果imu的队列不为空
            // pop and integrate imu data that is between two optimizations
            // 将队首的imu数据取出来
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            //时间上小于当前lidar位姿的imu数据都取出来
            if (imuTime < currentCorrectionTime - delta_t)
            {
                // 计算两个imu量之间的时间差
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);//这里处理了一下第一帧的情况，直接赋值为1.0 / 500.0
                // 调用预积分接口将imu数据送去处理
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                // 记录当前处理的imu数据的时间戳
                lastImuT_opt = imuTime;
                // 将这个数据从队列扔掉
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        // 两帧之间所有imu数据预计分完成之后，使用指针的转换工具，将其转换成预积分约束
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        // 预积分约束对相邻两帧之间的位姿，速度，零偏形成约束
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        //加入到因子图中
        graphFactors.add(imu_factor);
        // add imu bias between factor
        // 零偏约束， 两帧之间的零偏相差不会很大，因此使用常量约束
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        //将lidar的坐标转换到imu坐标系下
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        // 根据是否退化设置不同的置信度，作为这一帧的先验估计
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        // 加入到因子图中
        graphFactors.add(pose_factor);
        // insert predicted values
        // 根据上一时刻的状态，结合预积分结果，对当前状态进行预测。这里并没有直接使用lidarPose作为预测值
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // 预测量作为初始值插入因子图中
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);//零偏直接使用上一帧的零偏
        // optimize
        // 执行优化。连续执行两次
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        // 清零，方便进行下一步的操作
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        // 通过接口获取结果
        gtsam::Values result = optimizer.calculateEstimate();
        // 获取优化后的当前状态作为当前帧的最佳估计（位姿，速度，零偏）
        prevPose_  = result.at<gtsam::Pose3>(X(key));   // 位置
        prevVel_   = result.at<gtsam::Vector3>(V(key)); // 速度
        prevState_ = gtsam::NavState(prevPose_, prevVel_);  // 用位置和速度构造NavState类型的prevState_
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));   // 零偏
        // Reset the optimization preintegration object.
        // 当前的约束任务已经完成，预积分约束复位，同时需要设置一下零偏值，作为下一次积分的先决条件。（所以 gtsam中将预积分复位和设置零偏放到了一个函数）
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        // 一个简单的失败检测
        if (failureDetection(prevVel_, prevBias_))
        {
            // 状态异常直接复位
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 优化之后，根据最新的IMU状态进行传播
        // 前面通过lidar帧odom和imu预积分得到了这一lidar帧的更准确的位姿估计。这一过程耗时较长，已经有新的imu数据进了队列，
        // 为了保证odom的发布频率和imu频率相同，需要根据这一优化后的lidar帧位姿，结合新来的imu数据，推算新来的那几个imu的时刻的odom
        prevStateOdom = prevState_;//获得当前关键帧的位姿
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        // 把lidar帧之前的imu数据全部都弹出去
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        // 如果有新于lidar状态时刻的imu
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // 使用最新的零偏进行复位
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            // 以当前lidar帧的状态作为起始，把队列中剩下的imu数据开始积分
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);//时间戳
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);// 时间差
                // 将imu的数据加入到预积分中，注意这里并没有进行推算，也没有发布odom。因为发布的odom是以imu的频率发布的，所以推算和发布odom这一过程放在imu的回调中
                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    /**
     * @description: 对优化后的状态进行失败检查。失败的情况1、速度太快 2、零偏太大
     * @param {Vector3&} velCur
     * @param {ConstantBias&} biasCur
     * @return {*}
     */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            // 当速度太大，大于30m/s就认为是异常状态
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            // 零偏太大，也不正常
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }
    
    
    /**
     * @description: imu回调函数
     * @param {ConstPtr&} imu_raw 原始的imu数据
     * @return {*}
     */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);
        //首先把imu的数据进行一下简单的转换
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        //两个imu队列，作用不相同，一个用来执行与积分和位姿优化，一个用来更新最新imu状态
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        //如果没有发生过优化就return
        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        // 每来一个imu就家加入到预积分状态中
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        // 根据这个值预测最新的状态
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);// 这两个值都在lidar odom的回调函数odometryHandler中

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        // 转换格式，将状态转换到lidar坐标系下发送出去
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
