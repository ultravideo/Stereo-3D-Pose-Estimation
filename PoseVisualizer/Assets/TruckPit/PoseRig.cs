using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public struct posedata
{
    public int pose_id;
    public Vector3[] jointPositions; // = new Vector3[18];
    public Vector3[] jointPreviousPositions; // = new Vector3[18];
    public Vector3 headLookDir; // = Vector3.zero;
    public Vector3 midPelvis_p; // = Vector3.zero;
    public Vector3 pose_position;
    public GameObject[] jointGameobjects;

    public void clone_poses()
    {
        if (jointPositions != null)
            jointPreviousPositions = (Vector3[])jointPositions.Clone();
    }
};

public class PoseRig : MonoBehaviour
{
#region misc
    [SerializeField] Vector3 posePosition = new Vector3(0f, 10f, 0f);
    
    [SerializeField] float headBackOffset = 0.5f;
    [SerializeField] float poseScalingFactor = 3f;   
    [SerializeField] bool keepFeetOnGround = true;
    [SerializeField] Vector3 groundOffsetPlane = Vector3.zero;
    [SerializeField] bool timeSmoothPose = true;
    [SerializeField] bool disableLegs = true;
    [SerializeField] bool drawStickFigure = true;

    [SerializeField] GameObject[] jointGameobjects;
    // List<posedata> poses = new List<posedata>();
    public posedata pdata;

#endregion

#region body parts

    int REye_i = 14;
    int LEye_i = 15;
    int REar_i = 16;
    int LEar_i = 17;

    int midPelvis_i = 18;

    int Head_i = 0;

    int RHand_i = 4;
    int RElbow_i = 3;
    int RShoulder_i = 2;

    int LHand_i = 7;
    int LElbow_i = 6;
    int LShoulder_i = 5;

    int Chest_i = 1;

    int RPelvis_i = 8;
    int RKnee_i = 9;
    int RFoot_i = 10;
    int LPelvis_i = 11;
    int LKnee_i = 12;
    int LFoot_i = 13;

#endregion

#region max distances

    float HeadChest = 0.20f;
    float ChestShoulder = 0.20f;
    float ShoulderElbow = 0.302f;
    float ElbowHand = 0.269f;
    float ChestMidPelvis = 0.488f;
    float ChestPelvisDiagonal = 0.508f;
    float PelvisDiv2 = 0.14f;
    float PelvisKnee = 0.46f;
    float KneeFoot = 0.45f;

#endregion

    void Start()
    {
        pdata.jointPositions = new Vector3[18];
        pdata.jointGameobjects = jointGameobjects;
        pdata.pose_position = posePosition;
    }

    void Update()
    {
        if (timeSmoothPose)
            pdata = SmoothPose(pdata);

        pdata = NormalizeBoneLengths(pdata);

        pdata = ApplyJoints(pdata);
        pdata.clone_poses();
    }


    void DisableOutOfBounds()
    {
        foreach (GameObject t in jointGameobjects)
        {
            if (t == null)
                continue;

            if (t.transform.localPosition == posePosition)
                t.SetActive(false);
            else
                t.SetActive(true);
        }

        jointGameobjects[midPelvis_i].SetActive(true);
    }

    posedata SmoothPose(posedata pose)
    {
        if (pose.jointPositions == null || pose.jointPreviousPositions == null)
            return pose;

        for (int i = 0; i < pose.jointPositions.Length; i++)
        {
            pose.jointPositions[i] = pose.jointPositions[i] * 0.7f + pose.jointPreviousPositions[i] * 0.3f;
        }

        // pose.midPelvis_p = (pose.jointPositions[RPelvis_i] + pose.jointPositions[LPelvis_i]) / 2.0f;
        return pose;
    }

    posedata ScalePose(float scale_factor, posedata pose)
    {
        for (int i = 0; i < pose.jointPositions.Length; i++)
            pose.jointPositions[i] = pose.jointPositions[i] * scale_factor + pose.pose_position;

        pose.midPelvis_p = (pose.jointPositions[LPelvis_i] + pose.jointPositions[RPelvis_i]) / 2.0f;

        return pose;
    }

    posedata NormalizeBoneLengths(posedata pose)
    {
        pose.pose_position = (pose.jointPositions[RPelvis_i] + pose.jointPositions[LPelvis_i]) / 2.0f;

        Vector3[] jointPositions_ref = (Vector3[])pose.jointPositions.Clone();

        pose.midPelvis_p = pose.pose_position;

        // Debug.Log(midPelvis_ref.ToString("F4"));

        pose.jointPositions[RPelvis_i] = pose.midPelvis_p + (jointPositions_ref[RPelvis_i] - pose.midPelvis_p).normalized * PelvisDiv2;
        pose.jointPositions[LPelvis_i] = pose.midPelvis_p + (jointPositions_ref[LPelvis_i] - pose.midPelvis_p).normalized * PelvisDiv2;


        pose.jointPositions[Chest_i] = pose.midPelvis_p + (jointPositions_ref[Chest_i] - pose.midPelvis_p).normalized * ChestMidPelvis;

        pose.jointPositions[RShoulder_i] = pose.jointPositions[Chest_i] + (jointPositions_ref[RShoulder_i] - jointPositions_ref[Chest_i]).normalized * ChestShoulder;
        pose.jointPositions[LShoulder_i] = pose.jointPositions[Chest_i] + (jointPositions_ref[LShoulder_i] - jointPositions_ref[Chest_i]).normalized * ChestShoulder;

        pose.jointPositions[RElbow_i] = pose.jointPositions[RShoulder_i] + (jointPositions_ref[RElbow_i] - jointPositions_ref[RShoulder_i]).normalized * ShoulderElbow;
        pose.jointPositions[LElbow_i] = pose.jointPositions[LShoulder_i] + (jointPositions_ref[LElbow_i] - jointPositions_ref[LShoulder_i]).normalized * ShoulderElbow;

        pose.jointPositions[RHand_i] = pose.jointPositions[RElbow_i] + (jointPositions_ref[RHand_i] - jointPositions_ref[RElbow_i]).normalized * ElbowHand;
        pose.jointPositions[LHand_i] = pose.jointPositions[LElbow_i] + (jointPositions_ref[LHand_i] - jointPositions_ref[LElbow_i]).normalized * ElbowHand;

        pose.jointPositions[RKnee_i] = pose.jointPositions[RPelvis_i] + (jointPositions_ref[RKnee_i] - jointPositions_ref[RPelvis_i]).normalized * PelvisKnee;
        pose.jointPositions[LKnee_i] = pose.jointPositions[LPelvis_i] + (jointPositions_ref[LKnee_i] - jointPositions_ref[LPelvis_i]).normalized * PelvisKnee;

        pose.jointPositions[RFoot_i] = pose.jointPositions[RKnee_i] + (jointPositions_ref[RFoot_i] - jointPositions_ref[RKnee_i]).normalized * KneeFoot;
        pose.jointPositions[LFoot_i] = pose.jointPositions[LKnee_i] + (jointPositions_ref[LFoot_i] - jointPositions_ref[LKnee_i]).normalized * KneeFoot;

        pose.jointPositions[Head_i] = pose.jointPositions[Chest_i] + (jointPositions_ref[Head_i] - jointPositions_ref[Chest_i]).normalized * HeadChest;

        Vector3 pose_offset = Vector3.zero;
        if (keepFeetOnGround)
        {
            float offset = Mathf.Min(pose.jointPositions[LFoot_i].y, pose.jointPositions[RFoot_i].y);
            pose_offset = new Vector3(0f, -offset, 0f);
        }
        else
        {
            pose_offset = posePosition;
        }

        for (int i = 0; i < pose.jointPositions.Length; i++)
            pose.jointPositions[i] = pose.jointPositions[i] * poseScalingFactor + pose_offset;

        pose.midPelvis_p = pose.midPelvis_p * poseScalingFactor + pose_offset;

        return pose;
    }

    posedata ApplyJoints(posedata pose)
    {
        // if (pose.midPelvis_p == null || pose.jointTransforms == null)
        //     return;

        // Vector3 midpelvis_p = jointPositions[RPelvis_i] * 0.5f + jointPositions[LPelvis_i] * 0.5f;
        // midPelvis_p = new Vector3(0f, 1.5f, 0f);
        pose.jointGameobjects[midPelvis_i].transform.localPosition = pose.midPelvis_p;
        pose.jointGameobjects[RPelvis_i].transform.localPosition = pose.jointPositions[RPelvis_i];
        pose.jointGameobjects[LPelvis_i].transform.localPosition = pose.jointPositions[LPelvis_i];

        pose.jointGameobjects[RKnee_i].transform.localPosition = pose.jointPositions[RKnee_i];
        pose.jointGameobjects[LKnee_i].transform.localPosition = pose.jointPositions[LKnee_i];

        pose.jointGameobjects[RFoot_i].transform.localPosition = pose.jointPositions[RFoot_i];
        pose.jointGameobjects[LFoot_i].transform.localPosition = pose.jointPositions[LFoot_i];

        pose.jointGameobjects[Chest_i].transform.localPosition = pose.jointPositions[Chest_i];

        pose.jointGameobjects[RShoulder_i].transform.localPosition = pose.jointPositions[RShoulder_i];
        pose.jointGameobjects[LShoulder_i].transform.localPosition = pose.jointPositions[LShoulder_i];

        pose.jointGameobjects[RElbow_i].transform.localPosition = pose.jointPositions[RElbow_i];
        pose.jointGameobjects[LElbow_i].transform.localPosition = pose.jointPositions[LElbow_i];
 

        pose.jointGameobjects[RHand_i].transform.localPosition = pose.jointPositions[RHand_i];
        pose.jointGameobjects[LHand_i].transform.localPosition = pose.jointPositions[LHand_i];

        pose.jointGameobjects[Head_i].transform.localPosition = pose.jointPositions[Head_i];

        // DisableOutOfBounds();
        if (drawStickFigure)
            DrawStickFigureCheckOOB(pose);

        return pose;
    }

    void DrawStickFigureCheckOOB(posedata pose)
    {
        Debug.DrawLine(pose.jointGameobjects[Head_i].transform.position, pose.jointGameobjects[Head_i].transform.position + pose.headLookDir * 0.4f, new Color(0f, 0f, 1f));

        if (pose.jointGameobjects[Chest_i].gameObject.activeSelf && pose.jointGameobjects[LShoulder_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[Chest_i].transform.position, pose.jointGameobjects[LShoulder_i].transform.position, new Color(1f, 0f, 0f));

        if (pose.jointGameobjects[Chest_i].gameObject.activeSelf && pose.jointGameobjects[RShoulder_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[Chest_i].transform.position, pose.jointGameobjects[RShoulder_i].transform.position, new Color(1f, 0f, 0f));

        if (pose.jointGameobjects[midPelvis_i].gameObject.activeSelf && pose.jointGameobjects[Chest_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[midPelvis_i].transform.position, pose.jointGameobjects[Chest_i].transform.position, new Color(1f, 0f, 0f));


        if (pose.jointGameobjects[RShoulder_i].gameObject.activeSelf && pose.jointGameobjects[RElbow_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[RShoulder_i].transform.position, pose.jointGameobjects[RElbow_i].transform.position, new Color(1f, 0f, 0f));

        if (pose.jointGameobjects[RElbow_i].gameObject.activeSelf && pose.jointGameobjects[RHand_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[RElbow_i].transform.position, pose.jointGameobjects[RHand_i].transform.position, new Color(1f, 0f, 0f));

        if (pose.jointGameobjects[LShoulder_i].gameObject.activeSelf && pose.jointGameobjects[LElbow_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[LShoulder_i].transform.position, pose.jointGameobjects[LElbow_i].transform.position, new Color(1f, 0f, 0f));

        if (pose.jointGameobjects[LHand_i].gameObject.activeSelf && pose.jointGameobjects[LElbow_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[LElbow_i].transform.position, pose.jointGameobjects[LHand_i].transform.position, new Color(1f, 0f, 0f));

        if (pose.jointGameobjects[Chest_i].gameObject.activeSelf && pose.jointGameobjects[Head_i].gameObject.activeSelf)
            Debug.DrawLine(pose.jointGameobjects[Chest_i].transform.position, pose.jointGameobjects[Head_i].transform.position, new Color(1f, 0f, 0f));

        if (!disableLegs)
        {
            if (pose.jointGameobjects[midPelvis_i].gameObject.activeSelf && pose.jointGameobjects[RPelvis_i].gameObject.activeSelf)
                Debug.DrawLine(pose.jointGameobjects[midPelvis_i].transform.position, pose.jointGameobjects[RPelvis_i].transform.position, new Color(1f, 0f, 0f));

            if (pose.jointGameobjects[RKnee_i].gameObject.activeSelf && pose.jointGameobjects[RPelvis_i].gameObject.activeSelf)
                Debug.DrawLine(pose.jointGameobjects[RPelvis_i].transform.position, pose.jointGameobjects[RKnee_i].transform.position, new Color(1f, 0f, 0f));

            if (pose.jointGameobjects[RKnee_i].gameObject.activeSelf && pose.jointGameobjects[RFoot_i].gameObject.activeSelf)
                Debug.DrawLine(pose.jointGameobjects[RKnee_i].transform.position, pose.jointGameobjects[RFoot_i].transform.position, new Color(1f, 0f, 0f));

            if (pose.jointGameobjects[midPelvis_i].gameObject.activeSelf && pose.jointGameobjects[LPelvis_i].gameObject.activeSelf)
                Debug.DrawLine(pose.jointGameobjects[midPelvis_i].transform.position, pose.jointGameobjects[LPelvis_i].transform.position, new Color(1f, 0f, 0f));

            if (pose.jointGameobjects[LKnee_i].gameObject.activeSelf && pose.jointGameobjects[LPelvis_i].gameObject.activeSelf)
                Debug.DrawLine(pose.jointGameobjects[LPelvis_i].transform.position, pose.jointGameobjects[LKnee_i].transform.position, new Color(1f, 0f, 0f));

            if (pose.jointGameobjects[LKnee_i].gameObject.activeSelf && pose.jointGameobjects[LFoot_i].gameObject.activeSelf)
                Debug.DrawLine(pose.jointGameobjects[LKnee_i].transform.position, pose.jointGameobjects[LFoot_i].transform.position, new Color(1f, 0f, 0f));
        }
    }
}
