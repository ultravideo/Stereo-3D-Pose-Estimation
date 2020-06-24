using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IKTargeter : MonoBehaviour
{

    [SerializeField] Transform RHand;
    [SerializeField] Transform RElbow;
    [SerializeField] Transform RShoulder;

    [SerializeField] Transform LHand;
    [SerializeField] Transform LElbow;
    [SerializeField] Transform LShoulder;

    [SerializeField] Transform Head;
    [SerializeField] Transform Pelvis;
    [SerializeField] Transform Chest;
    
    [SerializeField] Transform RPelvis;
    [SerializeField] Transform RKnee;
    [SerializeField] Transform RFoot;

    [SerializeField] Transform LPelvis;
    [SerializeField] Transform LKnee;
    [SerializeField] Transform LFoot;

    [SerializeField] Transform SelfRHand;
    [SerializeField] Transform SelfRElbow;
    [SerializeField] Transform SelfRShoulder;

    [SerializeField] Transform SelfLHand;
    [SerializeField] Transform SelfLElbow;
    [SerializeField] Transform SelfLShoulder;

    [SerializeField] Transform SelfHead;
    [SerializeField] Transform SelfPelvis;
    [SerializeField] Transform SelfChest;

    [SerializeField] Transform SelfRPelvis;
    [SerializeField] Transform SelfRKnee;
    [SerializeField] Transform SelfRFoot;

    [SerializeField] Transform SelfLPelvis;
    [SerializeField] Transform SelfLKnee;
    [SerializeField] Transform SelfLFoot;


    void Update()
    {
        CalculateAngles();
    }

    void CalculateAngles()
    {

        

        // ??
        //Vector3 BackForthChestAngle = Vector3.ProjectOnPlane(Chest.position - Pelvis.position, Vector3.right);
        //float PelvisChestAngle = Vector3.Angle(Vector3.up, BackForthChestAngle);

        //float RElbowAngle = GetAngleBetween(RShoulder, RElbow, RHand);
    }

    float GetAngleBetween(Transform parent, Transform target, Transform child)
    {
        return Vector3.Angle(target.position - parent.position, child.position - target.position);
    }
}
