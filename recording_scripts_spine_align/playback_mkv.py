from argparse import ArgumentParser
import cv2
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def play(playback: PyK4APlayback):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    first = True
    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None and capture.depth is not None:
                capture._color = cv2.cvtColor(cv2.imdecode(
                    capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
                capture._color_format = ImageFormat.COLOR_BGRA32
                points = capture.transformed_depth_point_cloud.reshape(
                    (-1, 3)).astype('float64')
                colors = capture.color[..., (2, 1, 0)].reshape(
                    (-1, 3))
                geometry.points = o3d.utility.Vector3dVector(points)
                geometry.colors = o3d.utility.Vector3dVector(
                    (colors/255).astype('float64'))
                if first:
                    vis.add_geometry(geometry)
                    first = False
                else:
                    vis.update_geometry(geometry)
                vis.poll_events()
                vis.update_renderer()

        except EOFError:
            break


def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument(
        "--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
    parser.add_argument(
        "FILE", type=str, help="Path to MKV file written by k4arecorder")

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    play(playback)

    playback.close()


if __name__ == "__main__":
    main()
