using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class IKSolver : MonoBehaviour
{
    [SerializeField] private Transform Head;
    [SerializeField] private Transform LHand;
    [SerializeField] private Transform RHand;
    [SerializeField] private Transform LBow;
    [SerializeField] private Transform RBow;
    [SerializeField] private Transform LDer;
    [SerializeField] private Transform RDer;

    [SerializeField] private float HandShoulderMaxDistance = 0.54f;
    [SerializeField] private float HandElbowMaxDistance = 0.35f;
    [SerializeField] private float ElbowShoulderMaxDistance = 0.35f; //0.26f;
    [SerializeField] private float HeadShoulderYHeight = 0.18f;
    [SerializeField] private float HeadShoulderDiagonal = 0.25f;
    [SerializeField] private float HandWristOffset = -0.1f;


    void Start()
    {
        
    }

    void Update()
    {
        CalcShoulders();
        CalcElbows();
    }

    private void CalcElbows()
    {
        // calculate Hand positions with controller offsets
        Vector3 LHandPos = InputTracking.GetLocalPosition(XRNode.LeftHand);
        LHand.localPosition = LHandPos;
        LHand.localRotation = InputTracking.GetLocalRotation(XRNode.LeftHand);
        LHand.localPosition = LHandPos + LHand.forward * HandWristOffset;

        Vector3 RHandPos = InputTracking.GetLocalPosition(XRNode.RightHand);
        RHand.localPosition = RHandPos;
        RHand.localRotation = InputTracking.GetLocalRotation(XRNode.RightHand);
        RHand.localPosition = RHandPos + RHand.forward * HandWristOffset;

        // calculate ShoulderHand - ShoulderElbow angles 
        float currentLHandShoulderDistance = Vector3.Distance(LHand.position, LDer.position);
        float LAngle = -Mathf.Rad2Deg * Mathf.Acos((Mathf.Pow(HandElbowMaxDistance, 2)
            - Mathf.Pow(ElbowShoulderMaxDistance, 2)
            - Mathf.Pow(currentLHandShoulderDistance, 2))
            / (-2f * ElbowShoulderMaxDistance * currentLHandShoulderDistance));

        Vector3 LNormal = Vector3.Cross(-LHand.forward, LHandPos - LDer.position).normalized;
        Vector3 RNormal = Vector3.Cross(-RHand.forward, RHandPos - RDer.position).normalized;

        float currentRHandShoulderDistance = Vector3.Distance(LHand.position, LDer.position);
        float RAngle = -Mathf.Rad2Deg * Mathf.Acos((Mathf.Pow(HandElbowMaxDistance, 2)
            - Mathf.Pow(ElbowShoulderMaxDistance, 2)
            - Mathf.Pow(currentRHandShoulderDistance, 2))
            / (-2f * ElbowShoulderMaxDistance * currentRHandShoulderDistance));

        // calculate ShoulderHand vectors
        Vector3 LShoulderHandVec = (LHand.position - LDer.position).normalized;
        Vector3 RShoulderHandVec = (RHand.position - RDer.position).normalized;

        // calculate ShoulderElbow vector
        Vector3 LShoulderHand = (Quaternion.AngleAxis(LAngle, LNormal) * LShoulderHandVec).normalized;
        Vector3 RShoulderHand = (Quaternion.AngleAxis(RAngle, RNormal) * RShoulderHandVec).normalized;

        // calculate and set Elbow positions
        LBow.position = LDer.position + LShoulderHand * ElbowShoulderMaxDistance;
        RBow.position = RDer.position + RShoulderHand * ElbowShoulderMaxDistance;

    }
    
    private void CalcShoulders()
    {
        Vector3 belowHeadPos = Head.position - Vector3.up * HeadShoulderYHeight;
        float XDiff = Mathf.Sqrt(Mathf.Pow(HeadShoulderDiagonal, 2) + Mathf.Pow(HeadShoulderYHeight, 2));
        Vector3 shoulderDirNormalized = Quaternion.Euler(0f, Head.eulerAngles.y, 0f) * Vector3.right;

        LDer.position = belowHeadPos - shoulderDirNormalized * XDiff;
        RDer.position = belowHeadPos + shoulderDirNormalized * XDiff;
    }
}
