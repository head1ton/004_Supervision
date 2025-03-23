import argparse
from typing import List, Tuple, Iterable, Dict, Set

import cv2
import numpy as np
import supervision as sv
from shapely.creation import polygons
from ultralytics import YOLO


COLORS = sv.ColorPalette.DEFAULT

ZONE_IN_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]

class DetectionsManager:
    def __init__(self) -> None:
        # tracker_id를 zone_id에 매핑
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        # zone_out_id와 zone_in_id에 따라 tracker_id를 저장
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        # 각 zone_in에 대해 탐지된 tracker_id를 tracker_id_to_zone_id에 저장
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                # tracker_id가 이미 존재 하지 않으면 zone_in_id로 설정
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        # zone_out에 대해 탐지된 tracker_id를 counts[zone_out_id][zone_in_id]에 저장
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                # tracker_id가 tracker_id_to_zone_id 에 존재하는 경우
                if tracker_id in self.tracker_id_to_zone_id:
                    # tracker_id에 해당하는 zone_in_id를 가져옴
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    # zone_out_id에 해당하는 딕셔너리가 없으면 생성
                    self.counts.setdefault(zone_out_id, {})
                    # zone_out_id에 해당하는 zone_in_id의 집합이 없으면 생성
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    # tracker_id를 zone_out_id와 zone_in_id에 해당하는 집합에 추가
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        # 모든 탐지된 객체에 대해 class_id를 업데이트
        if len(detections_all) > 0:
            # tracker_id를 이용해 class_id를 설정, 존재하지 않으면 -1 로 설정
            detections_all.class_id = np.vectorize(lambda x: self.tracker_id_to_zone_id.get(x, -1))(detections_all.tracker_id)
        else:
            # 탐지된 객체가 없으면 빈 배열로 설정
            detections_all.class_id = np.array([], dtype=int)
        # class_id가 -1 이 아닌 탐지된 객체만 반환
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors
        )
        for polygon in polygons
    ]

class VideoProcessor():
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.source_weights_path = source_weights_path
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)
        self.detections_manager = DetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        frame_count = 0
        for frame in frame_generator:
            # frame_count += 1
            # if frame_count % 3 != 0:
            #     continue

            process_frame = self.process_frame(frame)
            cv2.imshow("frame", process_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            # print(zone_in.__dict__)
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone_in.polygon,
                color=COLORS.colors[i],
            )
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone_out.polygon,
                color=COLORS.colors[i]
            )

        labels = [
            f"#{tracker_id}"
            for tracker_id in detections.tracker_id
        ]
        annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame):
        result = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        print(self.detections_manager.counts)

        return self.annotate_frame(frame=frame, detections=detections)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Traffic Flow Analysis with YOLO and ByteTrack"
    # )
    #
    # parser.add_argument(
    #     "--source_weights_path",
    #     required=True,
    #     help="Path to the source weights file",
    #     type=str,
    # )
    #
    # parser.add_argument(
    #     "--source_video_path",
    #     required=True,
    #     help="Path to the source video file",
    #     type=str,
    # )
    #
    # parser.add_argument(
    #     "--target_video_path",
    #     default=None,
    #     help="Path to the target video file (output)",
    #     type=str,
    # )
    #
    # parser.add_argument(
    #     "--confidence_threshold",
    #     default=0.3,
    #     help="Confidence threshold for the model",
    #     type=float,
    # )
    #
    # parser.add_argument(
    #     "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    # )
    #
    # args = parser.parse_args()
    #
    # processor = VideoProcessor(
    #     source_weights_path=args.source_weights_path,
    #     source_video_path=args.source_video_path,
    #     target_video_path=args.target_video_path,
    #     confidence_threshold=args.confidence_threshold,
    #     iou_threshold=args.iou_threshold,
    # )

    processor = VideoProcessor(
        source_weights_path="data/traffic_analysis.pt",
        source_video_path="data/traffic_analysis.mov",
        target_video_path="data/traffic_analysis_result.mp4",
        confidence_threshold=0.3,
        iou_threshold=0.5,  # default values are used if not provided in command line arguments. 0.7 is a reasonable default for traffic flow analysis. 0.5 might be too low for other use cases. 0.9 is too high.
    )
    processor.process_video()

