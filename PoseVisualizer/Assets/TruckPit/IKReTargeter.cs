using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IKReTargeter : MonoBehaviour
{
    Animator animator;

    [SerializeField] Transform RootTransform;

    [SerializeField] Transform Head;

    [SerializeField] Transform RHand;
    [SerializeField] Transform RElbow;

    [SerializeField] Transform LHand;
    [SerializeField] Transform LElbow;

    [SerializeField] Transform RFoot;
    [SerializeField] Transform RKnee;

    [SerializeField] Transform LFoot;
    [SerializeField] Transform LKnee;

    [SerializeField] Transform SelfPelvis;
    [SerializeField] Transform SelfRFoot;

    float centerPivotDelta = 0;

    void Start()
    {
        animator = GetComponent<Animator>();
    }
    private void Awake()
    {
        centerPivotDelta = SelfPelvis.position.y - SelfRFoot.position.y;
    }

    private void Update()
    {
        transform.position = RootTransform.position - new Vector3(0, centerPivotDelta, 0);
    }

    private void OnAnimatorIK(int layerIndex)
    {
        if (!animator)
            return;

        animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 1);
        animator.SetIKPosition(AvatarIKGoal.RightHand, RHand.position);

        animator.SetIKHintPositionWeight(AvatarIKHint.RightElbow, 1);
        animator.SetIKHintPosition(AvatarIKHint.RightElbow, RElbow.position);


        animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, 1);
        animator.SetIKPosition(AvatarIKGoal.LeftHand, LHand.position);

        animator.SetIKHintPositionWeight(AvatarIKHint.LeftElbow, 1);
        animator.SetIKHintPosition(AvatarIKHint.LeftElbow, LElbow.position);


        animator.SetIKPositionWeight(AvatarIKGoal.RightFoot, 1);
        animator.SetIKPosition(AvatarIKGoal.RightFoot, RFoot.position);

        animator.SetIKHintPositionWeight(AvatarIKHint.RightKnee, 1);
        animator.SetIKHintPosition(AvatarIKHint.RightKnee, RKnee.position);


        animator.SetIKPositionWeight(AvatarIKGoal.LeftFoot, 1);
        animator.SetIKPosition(AvatarIKGoal.LeftFoot, LFoot.position);

        animator.SetIKHintPositionWeight(AvatarIKHint.LeftKnee, 1);
        animator.SetIKHintPosition(AvatarIKHint.LeftKnee, LKnee.position);

        animator.SetLookAtWeight(1);
        animator.SetLookAtPosition(Head.position - Head.forward);

        //animator.rootPosition = RootTransform.position;

    }


}
