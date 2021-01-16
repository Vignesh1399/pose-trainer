import argparse
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import glob
import json

class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))
        
        # normalize poses based on the average torso pixel length
        torso_lengths = np.array([Part.dist(pose.neck, pose.lhip) for pose in self.poses if pose.neck.exists and pose.lhip.exists] +
                                 [Part.dist(pose.neck, pose.rhip) for pose in self.poses if pose.neck.exists and pose.rhip.exists])
        mean_torso = np.mean(torso_lengths)

        for pose in self.poses:
            for attr, part in pose:
                setattr(pose, attr, part / mean_torso)



class Pose:
    PART_NAMES = ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'midhip', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear', 'lbigtoe', 'lsmalltoe' , 'lheel' , 'rbigtoe' , 'rsmalltoe' , 'rheel' , 'background']

    def __init__(self, parts):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 25 * 3 ndarray of x, y, confidence values
        """
        for name, vals in zip(self.PART_NAMES, parts):
            setattr(self, name, Part(vals))
    
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    
    def __str__(self):
        out = ""
        for name in self.PART_NAMES:
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out
    
    def print(self, parts):
        out = ""
        for name in parts:
            if not name in self.PART_NAMES:
                raise NameError(name)
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out

class Part:
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]
        self.exists = self.c != 0.0

    def __floordiv__(self, scalar):
        __truediv__(self, scalar)

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    @staticmethod
    def dist(part1, part2):
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))

def parse_sequence(json_folder, output_folder):
    """Parse a sequence of OpenPose JSON frames and saves a corresponding numpy file.

    Args:
        json_folder: path to the folder containing OpenPose JSON for one video.
        output_folder: path to save the numpy array files of keypoints.

    """
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    json_files = sorted(json_files)

    num_frames = len(json_files)
    #print('no of frames = ;',num_frames)
    all_keypoints = np.zeros((num_frames, 25, 3))
    for i in range(num_frames):
        with open(json_files[i]) as f:
            json_obj = json.load(f)
            keypoints = np.array(json_obj['people'][0]['pose_keypoints_2d'])
            #print(i)
            all_keypoints[i] = keypoints.reshape((25, 3))
    #print(all_keypoints)
    
    output_dir = os.path.join(output_folder, 'keypoints')
    #print('output_dir',output_dir)
    #print(os.path.basename(json_folder))
    np.save(output_dir, all_keypoints)


def load_ps(filename):
    """Load a PoseSequence object from a given numpy file.

    Args:
        filename: file name of the numpy file containing keypoints.
    
    Returns:
        PoseSequence object with normalized joint keypoints.
    """
    all_keypoints = np.load(filename)
    return PoseSequence(all_keypoints)

def evaluate_pose(pose_seq, exercise):
    """Evaluate a pose sequence for a particular exercise.

    Args:
        pose_seq: PoseSequence object.
        exercise: String name of the exercise to evaluate.

    Returns:
        correct: Bool whether exercise was performed correctly.
        feedback: Feedback string.

    """
    if exercise == 'squat':
        return _squat(pose_seq)
    else:
        return (False, "Exercise string not recognized.")


def _squat(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    #print(poses)
    right_present = [1 for pose in poses 
            if pose.rhip.exists and pose.rknee.exists and pose.rankle.exists]
    left_present = [1 for pose in poses
            if pose.lhip.exists and pose.lknee.exists and pose.lankle.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise leg detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rhip, pose.rknee, pose.rankle, pose.neck) for pose in poses]
    else:
        joints = [(pose.lhip, pose.lknee, pose.lankle, pose.neck) for pose in poses]
    #print(joints)

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]

    upper_leg_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    torso_vecs = np.array([(joint[3].x - joint[0].x, joint[3].y - joint[0].y) for joint in joints])
    lower_leg_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])

    # normalize vectors
    upper_leg_vecs = upper_leg_vecs / np.expand_dims(np.linalg.norm(upper_leg_vecs, axis=1), axis=1)
    torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
    lower_leg_vecs = lower_leg_vecs / np.expand_dims(np.linalg.norm(lower_leg_vecs, axis=1), axis=1)

    upper_leg_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_leg_vecs, torso_vecs), axis=1), -1.0, 1.0)))
    upper_leg_lower_leg_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_leg_vecs, lower_leg_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    upper_leg_torso_range = np.max(upper_leg_torso_angles) - np.min(upper_leg_torso_angles)
    upper_leg_lower_leg_min = np.min(upper_leg_lower_leg_angles)

    print('Upper leg and torso angle range: {}'.format(upper_leg_torso_range))
    print('Upper leg and lower leg minimum angle: {}'.format(upper_leg_lower_leg_min))

    correct = True
    feedback = ''

    if upper_leg_torso_range > 110.0:
        correct = False
        feedback += 'Your upper arm shows significant rotation around the shoulder when curling. Try holding your upper arm still, parallel to your chest, ' + \
                    'and concentrate on rotating around your elbow only.\n'
    
    if upper_leg_lower_leg_min > 75.0:
        correct = False
        feedback += 'You are not curling the weight all the way to the top, up to your shoulders. Try to curl your arm completely so that your forearm is parallel with your torso. It may help to use lighter weight.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Weight was lifted fully up, and upper arm did not move significantly.')
    else:
        return (correct, feedback)



def main():
    parser = argparse.ArgumentParser(description='Pose Trainer')
    parser.add_argument('--mode', type=str, default='evaluate', help='Pose Trainer application mode')
    parser.add_argument('--display', type=int, default=1, help='display openpose video')
    parser.add_argument('--input_folder', type=str, default='videos', help='input folder for videos')
    parser.add_argument('--output_folder', type=str, default='poses', help='output folder for pose JSON')
    parser.add_argument('--video', type=str, help='input video filepath for evaluation')
    parser.add_argument('--file', type=str, help='input npy file for evaluation')
    parser.add_argument('--exercise', type=str, default='squat', help='exercise type to evaluate')

    args = parser.parse_args()

    if args.mode == 'batch_json':
        # read filenames from the videos directory
        videos = os.listdir(args.input_folder)

        # openpose requires running from its directory
        os.chdir('openpose')

        for video in videos:
            print('processing video file:' + video)
            video_path = os.path.join('..', args.input_folder, video)
            output_path = os.path.join('..', args.output_folder, os.path.splitext(video)[0])
            openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
            subprocess.call([openpose_path, 
                            '--video', video_path, 
                            '--write_keypoint_json', output_path])

    elif args.mode == 'evaluate':
        if args.video:
            print('processing video file...')
            video = os.path.basename(args.video)
            
            output_path = "./poses/"
            openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
            #os.chdir('openpose')
            #subprocess.call([openpose_path, 
            #                '--video', os.path.join('..', args.video), 
            #                '--write_keypoint_json', output_path])
            parse_sequence(output_path, '.')
            pose_seq = load_ps('keypoints.npy')
            #print('\n\n\n\n\n\n')
            #print('pose_seq' ,pose_seq)
            #print('\n\n\n\n\n\n')
            (correct, feedback) = evaluate_pose(pose_seq, args.exercise)
            if correct:
                print('Exercise performed correctly!')
            else:
                print('Exercise could be improved:')
            print(feedback)
        else:
            print('No video file specified.')
            return
    
    elif args.mode == 'evaluate_npy':
        if args.file:
            pose_seq = load_ps(args.file)
            (correct, feedback) = evaluate_pose(pose_seq, args.exercise)
            if correct:
                print('Exercise performed correctly:')
            else:
                print('Exercise could be improved:')
            print(feedback)
        else:
            print('No npy file specified.')
            return
    
    else:
        print('Unrecognized mode option.')
        return




if __name__ == "__main__":
    main()
