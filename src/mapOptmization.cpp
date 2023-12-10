#include "utility.h"
#include <utility>
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

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

#include "jsk_recognition_msgs/BoundingBox.h"
#include "jsk_recognition_msgs/BoundingBoxArray.h"

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;//位姿协方差，前三维是旋转，后面三维是平移

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;
    
    ros::Publisher pubMatchedBBoxEdge;

    ros::Publisher pubbboxcur;
    ros::Publisher pubbboxlast;


    ros::Publisher pubSLAMInfo;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lio_sam::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;//关键帧的xyz信息
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    jsk_recognition_msgs::BoundingBoxArray BoxsCur;
    jsk_recognition_msgs::BoundingBoxArray BoxsCurTrans;
    jsk_recognition_msgs::BoundingBoxArray BoxsLast;
    std::vector<std::pair<jsk_recognition_msgs::BoundingBox, jsk_recognition_msgs::BoundingBox>> bbxmatched;

    std::vector<PointType> BboxVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelBboxVec;
    std::vector<bool> OriBboxFlag;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;

    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    // 上一帧优化后的最佳位姿，xyz roll pitch  yaw
    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;


    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        // 订阅cloud_info
        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅GPS
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        // 回环，可以手动的指定回环，人为指定有那两帧形成了回环
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);
        
        pubMatchedBBoxEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/matchbbox_constraints", 1);

        pubbboxcur = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/lio_sam/mapping/bboxcur", 1);
        pubbboxlast = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/lio_sam/mapping/bboxlast", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        pubSLAMInfo           = nh.advertise<lio_sam::cloud_info>("lio_sam/mapping/slam_info", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        BboxVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelBboxVec.resize(N_SCAN * Horizon_SCAN);
        OriBboxFlag.resize(N_SCAN * Horizon_SCAN);

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        // 提取时间戳
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();//转化成了以s做单位

        // extract info and feature cloud
        // 取出cloud_info中的角点和面点
        cloudInfo = *msgIn;
        BoxsCur = cloudInfo.BBoxArray; // 取出检测框
        BoxsCurTrans= cloudInfo.BBoxArray;

        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);   // 将角点转换为pcl格式
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);     // 面点点转换为pcl格式

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        // 控制后端的频率，配置文件中mappingProcessInterval是0.15s.雷达的帧率大概是0.1s，所以后端大概是每两帧处理一帧。这里可以根据算力调整
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;
            // 更新当前匹配结果的初始位姿
            updateInitialGuess();

            // 提取当前帧相关的关键帧并且构建点云地图
            extractSurroundingKeyFrames();

            // 对当前帧进行下采样
            downsampleCurrentScan();

            // 对点云配准进行优化问题构建求解 
            scan2MapOptimization();

            // 根据配准结果确定是否是关键帧
            saveKeyFramesAndFactor();

            // 调整全局的轨迹
            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    /**
     * @description: GPS消息的回调函数，将gpsMsg放入队列gpsQueue
     * @param {ConstPtr&} gpsMsg
     * @return {*}
     */
    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    /**
     * @description: 将点的坐标系从当前雷达帧的坐标系，通过当前帧的先验预测位姿转换到局部地图坐标系（世界坐标系下）
     * @param {PointType} *pi 输入点的坐标（在当前雷达帧的坐标系）
     * @param {PointType} *po 输出点的坐标（世界坐标系）
     * @return {*}
     */
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    void BoxsAssociateToMap(jsk_recognition_msgs::BoundingBoxArray BoxsCurori)
    {
        for (int i = 0; i < BoxsCurori.boxes.size(); i++)
        {

            //取出这个检测框的中心坐标
            Eigen::Vector3f BBboxTin(BoxsCurori.boxes[i].pose.position.x,
                                    BoxsCurori.boxes[i].pose.position.y,
                                    BoxsCurori.boxes[i].pose.position.z);

            // 取出旋转
            Eigen::Quaternionf BBboxquaterIn;
            BBboxquaterIn.x() = BoxsCurori.boxes[i].pose.orientation.x;
            BBboxquaterIn.y() = BoxsCurori.boxes[i].pose.orientation.y;
            BBboxquaterIn.z() = BoxsCurori.boxes[i].pose.orientation.z;
            BBboxquaterIn.w() = BoxsCurori.boxes[i].pose.orientation.w;
            
            // Eigen::Vector3f BBboxTout;
            Eigen::Vector3f BBboxTout =  transPointAssociateToMap * BBboxTin;

            Eigen::Quaternionf BBboxquaterOut =  Eigen::Quaternionf(transPointAssociateToMap.rotation())*BBboxquaterIn  ;
            
            // BBboxTout.x() = transPointAssociateToMap(0,0) * BBboxTin.x() + transPointAssociateToMap(0,1) * BBboxTin.y()  + transPointAssociateToMap(0,2) * BBboxTin.z()  + transPointAssociateToMap(0,3);
            // BBboxTout.y() = transPointAssociateToMap(1,0) * BBboxTin.x() + transPointAssociateToMap(1,1) * BBboxTin.y()  + transPointAssociateToMap(1,2) * BBboxTin.z()  + transPointAssociateToMap(1,3);
            // BBboxTout.z() = transPointAssociateToMap(2,0) * BBboxTin.x() + transPointAssociateToMap(2,1) * BBboxTin.y()  + transPointAssociateToMap(2,2) * BBboxTin.z()  + transPointAssociateToMap(2,3);
            
            // 重新赋值
            BoxsCurTrans.boxes[i].header.frame_id = "map";
            BoxsCurTrans.boxes[i].pose.position.x = BBboxTout.x();
            BoxsCurTrans.boxes[i].pose.position.y = BBboxTout.y();
            BoxsCurTrans.boxes[i].pose.position.z = BBboxTout.z();
            BoxsCurTrans.boxes[i].pose.orientation.x = BBboxquaterOut.x();
            BoxsCurTrans.boxes[i].pose.orientation.y = BBboxquaterOut.y();
            BoxsCurTrans.boxes[i].pose.orientation.z = BBboxquaterOut.z();
            BoxsCurTrans.boxes[i].pose.orientation.w = BBboxquaterOut.w();
        }
        BoxsCurTrans.header.frame_id = "map";
        pubbboxcur.publish(BoxsCurTrans);
        pubbboxlast.publish(BoxsLast);

    }

    
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        // 使用openmp进行并行加速
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            //每个点都要经过 Rp+t的位姿转换
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    













    bool saveMapService(lio_sam::save_mapRequest& req, lio_sam::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        // save corner cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        lio_sam::save_mapRequest  req;
        lio_sam::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }










    /**
     * @description: 回环检测线程
     * @return {*}
     */
    void loopClosureThread()
    {
        // 如果不需要回环检测（纯里程计）那么就直接退出这个线程
        if (loopClosureEnableFlag == false)
            return;

        // 设置回环检测的频率，1hz
        ros::Rate rate(loopClosureFrequency);
        // 死循环
        while (ros::ok())
        {
            // 每次sleep 1s，减少cpu占用
            rate.sleep();
            // 执行回环检测
            performLoopClosure();
            // 可视化
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        // 如果历史帧是空的，就没法进行回环检测了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;// 这两个变量在主线程中也进行操作，为了避免冲突，就拷贝了一份出来
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        // 首先看一下外部通知的回环消息
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            // 然后根据里程计的距离来检测回环
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // 检测出回环之后，开始计算两帧之间的位姿变换
        // extract cloud
        // 先事先准备两个空的点云
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 取出来当前帧点云，转换到世界坐标系下
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 取出来回环候选帧左右两侧各25帧，构建局部地图。
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            //如果点云数量太少就算了，当前帧的点太少或者历史帧构成的局部地图点云少于1000
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                // 把局部地图发布出去，供rviz可视化使用
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        // 使用简单的icp来进行帧到局部地图的配准
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);//设置最大相关距离15m*2
        icp.setMaximumIterations(100);//最大的优化次数
        icp.setTransformationEpsilon(1e-6);//优化步长
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        // 设置两个点云
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // 执行点云配准
        icp.align(*unused_result);

        // 检查icp是否收敛且的分是否满足要求
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        // 把修正后的当前点云发布供可视化使用
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        // 获得两个点云之间的变换矩阵结果
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        // 找到没有执行回环的当前帧的位姿，这是个不准的
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        //使用icp的结果，补偿到当前关键帧的位姿上面，就得到了更为准确的位姿结果
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        // 将矫正后的位姿转换成平移+欧拉角
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        // 矫正后的点云位姿
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // 矫正之前的点云位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        // 使用icp的得分作为他们的约束的噪声项
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        // 将两帧索引，两帧相对位姿，和噪声作为回环约束送入队列
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    /**
     * @description: 根据里程计检测回环
     * @param {int} *latestID
     * @param {int} *closestID
     * @return {*}
     */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 检测最新帧是否和其他帧形成回环，所以后面一帧的id就是最后一个关键帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;// 取出来最新关键帧的索引
        int loopKeyPre = -1;

        // check loop constraint added before
        // 检查一下较晚帧是否和别的帧形成了回环，如果有就算了
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        // 把只包含关键帧位移信息的点云填充kdtree
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // 根据最后一个关键帧的位置，寻找离当前关键帧一定距离内的其他关键帧
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        // 遍历较近的候选关键帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // 必须时间上超过30s。才认为是一个有效的回环
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;//只要找到了一个就行了，直接break
                break;
            }
        }

        // 如果没有找到回环或者回环找到自己身上去了，就认为是此次回环寻找失败
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * @description: 检测是否有外部通知的回环消息，作者表示这个功能还没有使用过，可以忽略
     * @param {int} *latestID
     * @param {int} *closestID
     * @return {*}
     */
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * @description: 
     * @param {Ptr&} nearKeyframes 
     * @param {int&} key
     * @param {int&} searchNum 搜索范围
     * @return {*}
     */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        // nearKeyframes 本来就是空的。这一步实际上没必要
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        //searchNum 搜索范围
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            // 找到这个idx
            int keyNear = key + i;
            // 如果超出了范围就算了
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            // 否则就把边缘点和面点的点云转到世界坐标系下去
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        // 如果没有有效的点云就算了
        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        // 把点云下采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);//体素滤波
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    void visualizeMatchedBBox()
    {
        if(bbxmatched.size() != 0)
        {
            visualization_msgs::MarkerArray MatchMarkerArray;
            // nodes
            visualization_msgs::Marker MatchMarkerNode;
            MatchMarkerNode.header.frame_id = odometryFrame;
            MatchMarkerNode.header.stamp = timeLaserInfoStamp;
            MatchMarkerNode.action = visualization_msgs::Marker::ADD;
            MatchMarkerNode.type = visualization_msgs::Marker::SPHERE_LIST;
            MatchMarkerNode.ns = "matched_bboxes";
            MatchMarkerNode.id = 0;
            MatchMarkerNode.pose.orientation.w = 1;
            MatchMarkerNode.scale.x = 1.0; MatchMarkerNode.scale.y = 1.0; MatchMarkerNode.scale.z = 1.0; 
            MatchMarkerNode.color.r = 1.0; MatchMarkerNode.color.g = 0.0; MatchMarkerNode.color.b = 0.0;
            MatchMarkerNode.color.a = 1;
            // edges
            visualization_msgs::Marker MatchMarkerEdge;
            MatchMarkerEdge.header.frame_id = odometryFrame;
            MatchMarkerEdge.header.stamp = timeLaserInfoStamp;
            MatchMarkerEdge.action = visualization_msgs::Marker::ADD;
            MatchMarkerEdge.type = visualization_msgs::Marker::LINE_LIST;
            MatchMarkerEdge.ns = "matched_bboxes_edges";
            MatchMarkerEdge.id = 1;
            MatchMarkerEdge.pose.orientation.w = 1;
            MatchMarkerEdge.scale.x = 1.0;
            MatchMarkerEdge.color.r = 0.9; MatchMarkerEdge.color.g = 0.9; MatchMarkerEdge.color.b = 0;
            MatchMarkerEdge.color.a = 1;

            for (auto it = bbxmatched.begin(); it != bbxmatched.end(); ++it)
            {
                geometry_msgs::Point p;
                p.x = it->first.pose.position.x;
                p.y = it->first.pose.position.y;
                p.z = it->first.pose.position.z;
                MatchMarkerNode.points.push_back(p);
                MatchMarkerEdge.points.push_back(p);
                p.x = it->second.pose.position.x;
                p.y = it->second.pose.position.y;
                p.z = it->second.pose.position.z;
                MatchMarkerNode.points.push_back(p);
                MatchMarkerEdge.points.push_back(p);
            }

            MatchMarkerArray.markers.push_back(MatchMarkerNode);
            MatchMarkerArray.markers.push_back(MatchMarkerEdge);
            pubMatchedBBoxEdge.publish(MatchMarkerArray);            
        }

    }





    



    /**
     * @description: 为了优化问题的求解，需要一个较好的初值。一个好的初值可以防止陷入局部最有，并且加速收敛。得到的初值保存在transFinal中
     * @return {*}
     */
    void updateInitialGuess()
    {
        // save current transformation before any processing
        // 转换为Eigen::Affine3f数据类型，transformTobeMapped是上一帧优化后的最佳位姿
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        // cloudKeyPoses3D是关键帧的三维坐标，如果是空的，代表没有关键帧，也就是系统正在初始化。这是第一帧，要做一些初始化相关的操作。
        if (cloudKeyPoses3D->points.empty())
        {
            //初始的位姿由磁力计提供
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            // 无论是vio还是lio系统的不可观都是4自由度（平移+yaw）。这里虽然有磁力计将yaw对齐，但是也可以考虑不使用yaw
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;//在初始时，将yaw置为0

            //保存磁力计得到的位姿，平移置0
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        // 如果有预计分提供的里程计，就利用里程计计算初值（一开始（初始化未完成）肯定是没有办法提供里程计的，因为预计分节点也需要后端节点提供的雷达里程计）
        if (cloudInfo.odomAvailable == true)
        {
            // 将提供的初值转换为eigen的数据结构保存下来
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            // 这个标志位表示是否第一次收到了预计分里程计信息
            if (lastImuPreTransAvailable == false)
            {
                // 将当前的里程计结果记录下来
                lastImuPreTransformation = transBack;
                // 收到了第一个里程计数据之后，这个标志位就是true
                lastImuPreTransAvailable = true;
            } else {
                // 计算上一个里程计的结果和当前里程计结果之间的delta pose
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                //将这个增量加到上一帧最佳位姿上去，就是当前帧的位姿的一个先验估计
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 将eigen转换成欧拉角和平移的形式
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                //保存当前帧的位姿
                lastImuPreTransformation = transBack;

                //虽然有里程计的信息，仍然需要将磁力计得到的旋转保存下来
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        // 如果没有里程计信息，就只用磁力计的旋转信息来作为初值更新，因为单纯使用imu无法得到靠谱的平移信息，因此，平移直接置0
        if (cloudInfo.imuAvailable == true)
        {
            //处置的计算方式和上面相同，但是平移置为0
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    /**
     * @description: 提取当前帧相关的关键帧并且构建点云地图
     * @return {*}
     */
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;    // 保存kdtree提取出来的元素的索引
        std::vector<float> pointSearchSqDis;    //保存距离查询位置的距离的数组

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        // 根据最后一个关键帧的信息，提取半径在一定距离内（默认50m）的关键帧。因为当前帧一般离最后一个关键帧不远，所以就直接以最后一个关键帧的位置为圆心搜索了
        // cloudKeyPoses3D->back()最后一个关键帧的位置
        // surroundingKeyframeSearchRadius搜索的半径
        // pointSearchInd 所有查询到的关键帧的索引
        // pointSearchSqDis查询到的关键帧离圆心的距离
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        //根据查询结果，将这些点的位置全部存进一个点云结构中
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // 避免关键帧过多，做一个下采样
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        //确认每个下采样后的点的索引，就是用一个最近邻搜索，其索引复制给这给的那的intensity数据位
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        // 除了提取空间上可能存在共视的关键帧，也要提取一些时间上接近的关键帧
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            // 最近十秒的关键帧也保存下来
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        //根据筛选出来的点进行局部地图的构建
        extractCloud(surroundingKeyPosesDS);
    }

    /**
     * @description: 
     * @param {Ptr} cloudToExtract
     * @return {*}
     */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        // 分别用来存放角点和面点的局部地图
        // 每来了新的一帧就clear
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 简单校验一下关键帧位置不能离当前帧太远。这个实际上不太会触发。（只有10秒内的很远的关键帧才会触发）
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            // 提取出来的关键帧的索引
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            // 如果laserCloudMapContainer中找到了这个关键帧，就说明之前某次已经处理过一次这个关键帧，转换到了全局坐标系并存到了laserCloudMapContainer中。
            // 这样就可以直接从laserCloudMapContainer中取出来加到局部地图中
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                //如果这个关键帧之前没有被处理过，那就通过该帧对应的位姿，把该帧对应的位姿转换到世界坐标系下
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                // 点云转换之后，加入到局部地图中去
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                // 把转换后的面点和角点存进这个容器中，方便后续直接加入点云地图，避免点云转换的操作，节约时间
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        // 将提取的点云转换世界坐标系下后，避免点云过于密集，因此对面点和角点的局部地图做一个下采样
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        // 如果局部地图容量过大，那就clear一下，避免内存占用过大
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    /**
     * @description: // 提取当前帧相关的关键帧并且构建点云地图
     * @return {*}
     */
    void extractSurroundingKeyFrames()
    {
        // 如果当前还没有关键帧，就直接return
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        // 对当前帧的角点和面点进行下采样，也是为了减少计算量
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);//角点下采样
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);//面点下采样
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    /**
     * @description: 将当前帧的先验的位姿transformTobeMapped 转换为Eigen::Affine3f格式
     * @return {*}
     */
    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void BBboxOptimization()
    {
        //首先清空匹配对
        bbxmatched.clear();


        updatePointAssociateToMap();
        // 将所有检测框通过先验位姿转换到世界坐标系下
        BoxsAssociateToMap(BoxsCur);
        // 存一下共有的激光点有多少，用来计算权重
        vector<int> bbxmatchednum;

        if(BoxsLast.boxes.size()!= 0)
        {
            // 两帧的检测框之间进行匹配
            // 遍历当前帧的检测框
            for(int i=0;i<BoxsCurTrans.boxes.size();i++)
            {
                // 取出这个检测框内的点云
                pcl::CropBox<pcl::PointXYZI> cropbox;
                cropbox.setInputCloud(laserCloudCornerFromMapDS);

                Eigen::Vector4f downleft(
                BoxsCurTrans.boxes[i].pose.position.x + BoxsCurTrans.boxes[i].dimensions.x/2,
                BoxsCurTrans.boxes[i].pose.position.y + BoxsCurTrans.boxes[i].dimensions.y/2,
                BoxsCurTrans.boxes[i].pose.position.z + BoxsCurTrans.boxes[i].dimensions.z/2,
                1.0);
                
                Eigen::Vector4f topright(
                BoxsCurTrans.boxes[i].pose.position.x - BoxsCurTrans.boxes[i].dimensions.x/2,
                BoxsCurTrans.boxes[i].pose.position.y - BoxsCurTrans.boxes[i].dimensions.y/2,
                BoxsCurTrans.boxes[i].pose.position.z - BoxsCurTrans.boxes[i].dimensions.z/2,
                1.0);
                cropbox.setMin(topright); //设置最小点
                cropbox.setMax(downleft);//设置最大点

                pcl::Indices clipped;

                cropbox.filter(clipped);

                int inboxCur =  clipped.size();
                
                if(inboxCur == 0)
                    continue;

                    // 遍历上一帧的检测框
                    for(int j=0;j<BoxsLast.boxes.size();j++)
                    {
                        // 取出这个检测框内的点云
                        pcl::CropBox<pcl::PointXYZI> cropbox2;
                        cropbox2.setInputCloud(laserCloudCornerFromMapDS);

                        Eigen::Vector4f downleft(
                        BoxsLast.boxes[j].pose.position.x + BoxsLast.boxes[j].dimensions.x/2,
                        BoxsLast.boxes[j].pose.position.y + BoxsLast.boxes[j].dimensions.y/2,
                        BoxsLast.boxes[j].pose.position.z + BoxsLast.boxes[j].dimensions.z/2,
                        1.0);
                        
                        Eigen::Vector4f topright(
                        BoxsLast.boxes[j].pose.position.x - BoxsLast.boxes[j].dimensions.x/2,
                        BoxsLast.boxes[j].pose.position.y - BoxsLast.boxes[j].dimensions.y/2,
                        BoxsLast.boxes[j].pose.position.z - BoxsLast.boxes[j].dimensions.z/2,
                        1.0);
                        cropbox2.setMin(topright);   //设置最小点
                        cropbox2.setMax(downleft);   //设置最大点

                        pcl::Indices clipped2;

                        cropbox2.filter(clipped2);
                        
                        if(clipped2.size() == 0)
                            continue;

                        // 两个vector取交集
                        pcl::Indices both;
                        sort(clipped.begin(), clipped.end());
                        sort(clipped2.begin(), clipped2.end());
                        set_intersection(clipped.begin(), clipped.end(), clipped2.begin(), clipped2.end(), back_inserter(both));
                        
                        // 如果即在检测框1又在检测框2中的点云数量都大于各自的80%，就认为这两帧中的这两个检测框是一个物体
                        if(both.size()>=5 && both.size() >= 0.8*clipped.size() && both.size() >= 0.8*clipped2.size())
                        {                        
                            ROS_INFO("both size: %d  ,clipped.size : %d ,clipped2.size : %d ",both.size(),clipped.size(),clipped2.size());
                            // ROS_INFO("\033[1;32m get matched \033[0m");
                            //把匹配对添加到vector里面去
                            bbxmatched.push_back(std::make_pair(BoxsCurTrans.boxes[i],BoxsLast.boxes[j]));
                            bbxmatchednum.push_back(both.size());

                            //找到一个就行了，直接退出。继续找下一个当前帧的BoxsCur.boxes[i]
                            break;
                        }
                }

            }
            for (int i = 0; i < bbxmatched.size(); i++)
            {
                PointType coeff;
                PointType pointOri;
                pointOri.x = bbxmatched[i].first.pose.position.x;
                pointOri.y = bbxmatched[i].first.pose.position.y;
                pointOri.z = bbxmatched[i].first.pose.position.z;

                // 残差
                float ld2 = (bbxmatched[i].first.pose.position.x-bbxmatched[i].second.pose.position.x)*(bbxmatched[i].first.pose.position.x-bbxmatched[i].second.pose.position.x)+
                            (bbxmatched[i].first.pose.position.y-bbxmatched[i].second.pose.position.y)*(bbxmatched[i].first.pose.position.y-bbxmatched[i].second.pose.position.y)+
                            (bbxmatched[i].first.pose.position.z-bbxmatched[i].second.pose.position.z)*(bbxmatched[i].first.pose.position.z-bbxmatched[i].second.pose.position.z);
                float la = (bbxmatched[i].first.pose.position.x-bbxmatched[i].second.pose.position.x)/ld2;
                float lb = (bbxmatched[i].first.pose.position.y-bbxmatched[i].second.pose.position.y)/ld2;
                float lc = (bbxmatched[i].first.pose.position.z-bbxmatched[i].second.pose.position.z)/ld2;
                // 一个简单的核函数，残差越大权重越低
                // float s =  (1.0 - fabs(ld2)) * 0.05*float(bbxmatchednum[i]);
                float s =  (1.0 - fabs(ld2));
                // float s =  (1.0 - fabs(ld2)) * sigmoid(0.05*float(bbxmatchednum[i]));
                coeff.x = s * la;
                coeff.y = s * lb;
                coeff.z = s * lc;
                coeff.intensity = s * ld2;

                ROS_INFO("\033[1;32m ld2=%f s=%f  matchednum=%d \033[0m",ld2,s,bbxmatchednum[i]);
                if (s > 0.1) {
                    ROS_INFO("\033[1;32m match  good %d \033[0m",i);
                    BboxVec[i] = pointOri;
                    coeffSelBboxVec[i] = coeff;
                    OriBboxFlag[i] = true;
                }
            }

        }
    }

    // float sigmoid(float x)
    // {
    //     return (1.0 / (1.0 + exp(-x)));
    // }

    void cornerOptimization()
    {
        updatePointAssociateToMap();

        //使用openmp进行并行加速
        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历当前帧的所有角点
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];//取出这个点的坐标
            //将这个点，通过先验的位姿转换到局部地图坐标系下（局部地图坐标系就是全局世界坐标系）
            pointAssociateToMap(&pointOri, &pointSel);
            //进行最近临查找，在角点地图里面寻找距离当前点较近的五个点
            // pointSel 在pointSel周围搜索
            // 5 找五个
            // pointSearchInd 找到的点的索引
            // pointSearchSqDis 数组
            // 注意 PCL库寻找最近邻点的返回的结果中，是按照距离从小到大排序的
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            // 计算找到的地图点中距离当前点最远的点，如果距离体阿达就说明这个约束不太可信，就跳过
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;   //  五个最近邻点的中心位置
                // 计算协方差矩阵
                // 首先计算均值
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 协方差矩阵matA1
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                //特征值分解
                cv::eigen(matA1, matD1, matV1);

                //这是线特征，要求最大特征值大于三倍的次大特征值。如果不是，跳过
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 特征向量对应的就是直线的方向向量
                    // 通过五点的中心，沿着方向向量向两边移动，构成了可以确定一条线的两个端点
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 计算点到线的残差（垂线的长度）和垂线方向（以及雅可比的方向）
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    // 一个简单的核函数，残差越大权重越低
                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    // 如果残差小于10cm，就认为是一个有效的约束
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * @description: 边缘点优化，与角点优化函数cornerOptimization()类似
     * @return {*}
     */
    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 同样找5个面点
            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            // 平面方程 Ax+By+Cz+1=0
            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // 同样最大距离不能超过1m
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 求解Ax=B的超定方程
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // 求出来x就是这个平面的法向量
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                // 归一化，将法向量模长统一为1
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    // 每个点带入到平面方程，计算点到平面的距离，如果距离大于0.2m就认为这个平面的曲率过大，就是无效的平面
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                //如果是有效的平面
                if (planeValid) {
                    // 计算当前点到平面的距离
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    //分母的意义是？？？？
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }
  
    /**
     * @description:将角点和面点的约束统一到一起，将角点和面点的信息都放到laserCloudOri和coeffSel里面去。一维残差，三维雅可比，三维点坐标。这里已经不区分角点和面点了
     * @return {*}
     */
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            // 只有标志位为true的时候才是有效的约束
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
    
        // combine BBbox coeffs
        for (int i = 0; i < bbxmatched.size(); ++i){
            if (OriBboxFlag[i] == true){
                laserCloudOri->push_back(BboxVec[i]);
                coeffSel->push_back(coeffSelBboxVec[i]);
            }
        }
        // reset flag for next iteration
        // 标志位清零
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
        std::fill(OriBboxFlag.begin(), OriBboxFlag.end(), false);
    }

    /**
     * @description:  进行一次梯度下降优化。高斯牛顿法实现，并不涉及LM。原始的loam的代码时将lidar坐标系转换到相机坐标系。这部分代码是从loam里面拷贝过来了，为了坐标系的统一，就先和loam中一样，先转换到相机系优化，然后结果转换到lidar系中
     * @param {int} iterCount
     * @return {*}
     */
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        // 将lidar系转到相机系
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        // 如果约束太少，小于50就return
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            //首先将当前点以及点到线（面）的单位向量转换到相机系中
            // lidar -> camera
            // 点
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            // 雅可比（梯度下降的方向）
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 相机系下旋转顺序是Y-X-Z，对应lidar系下的ZYX
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // camera -> lidar
            // 这里是吧camera转到lidar了
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        // 构造JT*J以及JT*e矩阵
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // 求解增量
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            // 检查是否有退化的情况
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 对JT*J进行特征值分解
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                // 对特征值从小到大进行遍历，如果小于阈值就认为i是退化的
                if (matE.at<float>(0, i) < eignThre[i]) {
                    // 对应的特征向量全部置为0
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;// 退化标志为置为true
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        // 如果发生退化，就对增量进行修改，退化方向不更新
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 增量更新
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        //计算更新的旋转和平移的大小
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // 如果旋转和平移的增量足够小，认为优化问题收敛了
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        // 否则继续优化
        return false; // keep optimizing
    }

    /**
     * @description: 点云配准 当前帧和局部地图进行匹配， 求一个旋转和平移，使得当前帧能够最好得匹配到局部地图上面
     * @return {*}
     */
    void scan2MapOptimization()
    {
        //如果没有关键帧，那么就没有局部地图，也就没办法做当前帧到局部地图的匹配
        if (cloudKeyPoses3D->points.empty())
            return;

        // 判断当前帧的角点和面点数量是否足够
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 把局部地图的角点和面点构建kdtree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 迭代求解，一个简单的手写的高斯牛顿法。如果优化次数大于30或者收敛了，就退出
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();//边缘点的优化
                surfOptimization();//面点的优化
                BBboxOptimization();//获取检测框的匹配

                combineOptimizationCoeffs();

                // 进行一次优化，如果收敛了就可以结束了
                if (LMOptimization(iterCount) == true)
                    break;              
            }

            // 优化问题结束
            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    /**
     * @description: 把结果和imu进行一些加权融合
     * @return {*}
     */
    void transformUpdate()
    {
        // 是否可以获得九轴imu在世界坐标下的姿态
        if (cloudInfo.imuAvailable == true)
        {
            // 因为roll和pitch原则上全程可观，因此这里把lidar推算出来的姿态和磁力计结果做一个加权平均
            // 首先判断翻车了没有，如果翻车了好像做slam也没什么意义了。当然手持设备可以pitch很大，这里主要避免插值产生的奇异
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                // lidar点云匹配获得的roll角转换成四元数
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                // imu获得的roll角
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                // 使用四元数球面插值 
                // imuWeight权重。
                // rollMid = lidar * (1-imuWeight) + imu * imuWeight
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                // 插值结果作为roll的最终结果
                transformTobeMapped[0] = rollMid;

                // 下面pitch角同理
                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // 对roll，pitch和z进行一些玉树，主要针对室内2D场景下，已知2d的先验可以加上这些约束。z轴不能太高，roll和pitch也不会变化太大
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        // 最终的结果也可以转成eigen的结构
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    
        BoxsAssociateToMap(BoxsCur);
        BoxsLast = BoxsCurTrans; //记录检测框
        visualizeMatchedBBox();

    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    /**
     * @description: 判断当前帧是否是关键帧
     * @return {*}
     */
    bool saveFrame()
    {
        // 如果现在还没有关键帧（这是第一帧），那么就直接认为该帧就是关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        // 取出上一个关键帧的位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 当前帧的位姿（scan to map得到的），格式转成eigen格式
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 计算两个位姿之间的delta pose
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        // 转换成平移+旋转的形式
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 任何一个旋转大于给定的阈值或者平移大于给定的阈值的就认为是关键帧
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    /**
     * @description: 添加odom的因子
     * @return {*}
     */
    void addOdomFactor()
    {
        // 如果是第一帧（第一帧一定会被添加成关键帧），增加的是“先验约束”
        if (cloudKeyPoses3D->points.empty())
        {
            // 置信度设置的差一点，尤其是不可观的平移和yaw角。第一帧的平移设置为0， roll和pitch从imu获得，yaw一般设置为0（一般不做磁力计和系统初始yaw的对齐）
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 增加先验约束，对第0个节点增加约束
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 设置变量，加入节点信息
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            // 如果不是第一帧，就增加“帧间约束”
            // 帧间约束的置信度就设置的高一点
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // 转换成gtsam的格式
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 上一帧的索引是size-1。 当前帧索引正好是size
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 加入节点信息。当前帧的索引和先验位姿
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    /**
     * @description: 添加GPS的因子。gpsQueue队列中存储的GPS消息
     * @return {*}
     */
    void addGPSFactor()
    {
        // 如果没有gps信息就算了
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        // 如果没有关键帧，那也没用。GPS就是给关键帧提供约束的
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            // 第一个关键帧和最后一个关键帧距离很近也就算球，因为这种情况要么是刚刚起步，要么是会触发回环了
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // gtsam反馈的当前xy的置信度，如置信度比较高也不需要GPS来打扰。不考虑Z是因为，GPS的Z不准，所以也不需要考虑用对GPS对Z进行约束
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // 把距离当前帧比较早的GPS数据全都扔掉
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            //如果lidar的数据来的太慢了，GPS时间超前最新的lidar帧的时间。就等一等雷达
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                //GPS的再时间上距离当前帧已经比较近了（在+-0.2s的时间范围内），那么就把这个数据thisGPS取出来
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                // 把GPS的噪声取出来，
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                // 如果GPS的协方差太大了，就算了，置信度太差不能用
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                //取出GPS的位置xyz
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                // 是否使用GPS的z。通常是不用GPS的z的，因为没有xy准
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];//如果不使用GPS的z，就直接用lidar里程计的z作为gps_z
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                // GPS的xy太小了，说明还没有初始化完成
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                // 加入GPS不能太频繁，相邻不能超过5m
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                // gps的置信度，即便是前面有了协方差，但是这里依然要把协方差设置成最小1m，也就是不会特别信任GPS的信号
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                // 调用gtsam中集成的gps的约束。（先验约束）
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                //加入到因子图
                gtSAMgraph.add(gps_factor);

                // 注意，不需要调用insert加入新的变量了，因为GPS约束没有引入新的待优化变量进来，待优化变量都是关键帧的位姿，已经再addodomfactor里面添加了
                // 加入之后，等同于回环，需要触发大较多的isam update
                aLoopIsClosed = true;
                break;
            }
        }
    }

    /**
     * @description: 添加回环的因子
     * @return {*}
     */
    void addLoopFactor()
    {
        // 有一个专门的回环检测线程会检测回环，检测到就会给这个队列loopIndexQueue塞入回环结果
        // 如果没有检测到回环就算了
        if (loopIndexQueue.empty())
            return;

        // 把队列中所有的回环约束都添加进来
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 这是一个帧间约束
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // 回环的置信度就是icp的得分
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            // 加入约束
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        // 清空回环相关的队列
        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        //标志位 置true
        aLoopIsClosed = true;
    }

    /**
     * @description: 判断当前帧是否是关键帧，如果是添加三种因子，调用isam接口进行更新
     * @return {*}
     */
    void saveKeyFramesAndFactor()
    {
        // 当前帧是否是关键帧。如果不是，那就不会向因子图中添加关键帧等操作了
        if (saveFrame() == false)
            return;

        //如果是关键帧，那就向iSAM中增加因子，除了imu预计分的factor以外，其余的三个factor全都在这里被加了进来
        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // 所有的因子加完了，就调用isam的接口更新图模型
        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // 如果加入了gps的约束或者回环约束，isam需要进行更多次数的优化
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        // 将约束和节点信息清空，他们已经被加入到了isam中去了，因此这里清空不会影响到整个优化
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // 下面保存关键帧的信息
        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        // 取出优化后的最新关键帧的位姿
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // 平移信息取出来保存到cloudKeyPoses3D中去，其中索引作为intensity值
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        // 全部的位姿信息，包括时间存到cloudKeyPoses6D中，intensity为索引
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 保存当前位姿的置信度，用来判断是否要加GPS
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        // 将isam优化后的位姿返还给transformTobeMapped中，作为当前的最佳估计值
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        // 当前帧的点云的角点和面点分别拷贝一下。（下一帧来的时候thisCornerKeyFrame和thisSurfKeyFrame会被新的角点和面点覆盖）
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        // 关键帧的点云存起来
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // 根据当前位姿更新rviz可视化
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        // 没有关键帧自然也就没什么意义了
        if (cloudKeyPoses3D->points.empty())
            return;

        // 只有回环或者融入GPS信息的是才会触发全局路径点的调整
        if (aLoopIsClosed == true)
        {
            // clear map cache
            // 很多位姿都会发生变化，因此之前的容器内已经转换到世界坐标系下的很多点云都需要调整，因此这里直接清空
            laserCloudMapContainer.clear();
            // clear path
            // globalPath是可视化用的，也直接清空path
            globalPath.poses.clear();
            // update key poses
            //然后更新所有位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                // 更新所有关键帧的位姿
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                // 同时更新path
                updatePath(cloudKeyPoses6D->points[i]);
            }

            // 标志位置位
            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    /**
     * @description: 这里发布了两个里程计话题。
     * pubLaserOdometryGlobal发布的是回环优化后的odom数据；
     * pubLaserOdometryIncremental则是没有经过回环优化的里程计数据，这是给IMU预计分节点用的。回环的跳变对于imu预计分的因子图优化是灾难性的
     * 如果没有检测到闭环，则这两个是一样的。
     * @return {*}
     */
    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        // 发送当前帧的位姿
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        
        // Publish TF
        // 发送lidar再odom坐标系下的tf
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        // 发送增量位姿变换
        // 这里发送的位姿变换增量是给imu预计分节点用的，因为imu预计分因子图优化要求里程计必须是平滑的不能有回环的那种跳变，
        // 所以为了避免回环的里程计跳变带来的优化灾难，这里发布的是一个两帧之间位姿的增量
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        // 该标志位处理一次以后始终为true
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            // 记录当前位姿
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            // 上一帧的最佳位姿和当前帧的最佳位姿（scan match得到的而不是回环或者GPS之后得到的位姿）之间的位姿差。纯scan match得到的位姿之间的delta T
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            // 位姿的增量叠加到上一帧位姿上
            increOdomAffine = increOdomAffine * affineIncre;
            // 分解成欧拉角+平移向量的形式
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            // 如果有imu信号，同样对roll和pitch做插值
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            //协方差这一位作为是否退化的标志位
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    /**
     * @description: 发布可视化的点云信息
     * @return {*}
     */
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                lio_sam::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudCornerLastDS;
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudCornerFromMapDS;
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    // 两个线程
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);// 回环检测的线程
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);//可视化的线程，rviz可视化接口的发布

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
