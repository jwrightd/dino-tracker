import argparse
import os
import imageio
import cv2


def mp4_to_frames(mp4_file, output_folder, max_frames=None):
    os.makedirs(output_folder, exist_ok=True)
    try:
        vid = imageio.get_reader(mp4_file)
        for i, frame in enumerate(vid):
            if max_frames is not None and i >= max_frames:
                break
            imageio.imwrite(os.path.join(output_folder, f"{i:05d}.jpg"), frame)
        return
    except Exception:
        pass

    cap = cv2.VideoCapture(mp4_file)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_file}")
    try:
        i = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames is not None and i >= max_frames:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            imageio.imwrite(os.path.join(output_folder, f"{i:05d}.jpg"), frame_rgb)
            i += 1
    finally:
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str)
    parser.add_argument("--output-folder", type=str)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    mp4_to_frames(args.video_path, args.output_folder, max_frames=args.max_frames)

# python mp4_to_frames.py --video-path ./dataset/libby-test/visualizations/dotted_tracks_fps_10.mp4 --output-folder /dataset/libby-test/visualizations-frames
