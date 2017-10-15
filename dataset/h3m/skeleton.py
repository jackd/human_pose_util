from human_pose_util.skeleton import Skeleton, front_angle, skeleton_height

thorax = 'thorax'
r_hip = 'r_hip'
r_knee = 'r_knee'
r_ankle = 'r_ankle'
r_ball = 'r_ball'
r_toes = 'r_toes'
l_hip = 'l_hip'
l_knee = 'l_knee'
l_ankle = 'l_ankle'
l_ball = 'l_ball'
l_toes = 'l_toes'
neck_base = 'neck'
head_center = 'head-center'
head_back = 'head-back'
l_shoulder = 'l_shoulder'
l_elbow = 'l_elbow'
l_wrist = 'l_wrist'
l_thumb = 'l_thumb'
l_little = 'l_little'
r_shoulder = 'r_shoulder'
r_elbow = 'r_elbow'
r_wrist = 'r_wrist'
r_thumb = 'r_thumb'
r_little = 'r_little'
# pelvis = 'pelvis'


class _S24(Skeleton):
    def __init__(self):
        links = (
            (thorax, None),
            (r_hip, thorax),
            # (r_hip, pelvis),
            (r_knee, r_hip),
            (r_ankle, r_knee),
            (r_ball, r_ankle),
            (r_toes, r_ball),
            (l_hip, thorax),
            # (l_hip, pelvis),
            (l_knee, l_hip),
            (l_ankle, l_knee),
            (l_ball, l_ankle),
            (l_toes, l_ball),
            (neck_base, thorax),
            # (head_center, head_back),
            # (head_back, neck_base),
            # (head_back, head_center),
            # (head_center, neck_base),
            (head_back, neck_base),
            (head_center, head_back),

            (l_shoulder, neck_base),
            (l_elbow, l_shoulder),
            (l_wrist, l_elbow),
            (l_thumb, l_wrist),
            (l_little, l_wrist),
            (r_shoulder, neck_base),
            (r_elbow, r_shoulder),
            (r_wrist, r_elbow),
            (r_thumb, r_wrist),
            (r_little, r_wrist),
            # (pelvis, thorax),
        )
        assert(len(links) == 24)
        super(_S24, self).__init__(links)

    def height(self, p3):
        return skeleton_height(
            p3, l_foot_index=self.joint_index(l_toes),
            r_foot_index=self.joint_index(r_toes),
            head_index=self.joint_index(head_back))

    def front_angle(self, p3):
        return front_angle(self, p3, l_joint=l_hip, r_joint=r_hip)


s24 = _S24()
