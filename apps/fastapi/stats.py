from pydantic import BaseModel
from ..nn.yolov8.score import YOLOv8Objects, analyze_video
import numpy as np
from _collections_abc import Sequence

class ScoreCardReport(BaseModel):
    # https://docs.google.com/spreadsheets/d/1QT3j336EuLBRHXdh-p6GNt12al1fo9Lt/edit#gid=316097104

    allButCommonRawFloor: float | None
    allButCommonIntermediateFloor: float | None
    allButCommonFinishedFloor: float | None

    allButCommonRawWall: float | None
    allButCommonStartedWall: float | None
    allButCommonFinishedWall: float | None

    allButCommonRawCeiling: float | None
    # allButCommonStartedCeiling: float | None  # no such class in terms of reference
    allButCommonFinishedCeiling: float | None

    allDoors: float | None
    allGarbage: int | None
    allSocketsAndSwitches: int | None

    livingOrKitchenWindow: float | None
    livingOrKitchenRadiator: float | None
    livingOrKitchenFurniture: int | None

    bathroomToiletSeat: float | None
    bathroomToiletBathtub: float | None
    bathroomToiletSink: float | None

    commonAreasRawFloor: float | None
    commonAreasIntermediateFloor: float | None
    commonAreasFinishedFloor: float | None

    commonAreasRawWall: float | None
    commonAreasStartedWall: float | None
    commonAreasFinishedWall: float | None

    commonAreasRawCeiling: float | None
    # commonAreasStartedCeiling: float | None  # no such class in terms of reference
    commonAreasFinishedCeiling: float | None

    final_score: float | None


def derive_statistics(outputs: list[YOLOv8Objects]) -> ScoreCardReport:
    """Given a list of YOLOv8Objects, derive statistics for each object.
    """
    stats = ScoreCardReport()
    stats.allGarbage = len([obj for obj in outputs if obj.className == 'garbage'])
    stats.allSocketsAndSwitches = len([obj for obj in outputs if obj.className == 'socket' or obj.className == 'switch'])

    number_of_bathrooms = len([obj for obj in outputs if obj.roomClass == 'bathroom'])
    bathrooms_with_bathub = len([obj for obj in outputs if obj.roomClass == 'bathroom' and obj.className == 'bathtub'])
    bathrooms_with_toilet = len([obj for obj in outputs if obj.roomClass == 'bathroom' and obj.className == 'toilet seat'])
    bathrooms_with_sink = len([obj for obj in outputs if obj.roomClass == 'bathroom' and obj.className == 'sink'])
    stats.bathroomToiletBathtub = bathrooms_with_bathub / number_of_bathrooms if number_of_bathrooms > 0 else 0
    stats.bathroomToiletSeat = bathrooms_with_toilet / number_of_bathrooms if number_of_bathrooms > 0 else 0
    stats.bathroomToiletSink = bathrooms_with_sink / number_of_bathrooms if number_of_bathrooms > 0 else 0

    number_of_livingrooms = len([obj for obj in outputs if obj.roomClass == 'livingroom'])
    livingrooms_with_radiator = len([obj for obj in outputs if obj.roomClass == 'livingroom' and obj.className == 'radiator'])
    livingrooms_with_furniture = len([obj for obj in outputs if obj.roomClass == 'livingroom' and obj.className == 'kitchen'])
    stats.livingOrKitchenRadiator = livingrooms_with_radiator / number_of_livingrooms if number_of_livingrooms > 0 else 0
    stats.livingOrKitchenFurniture = livingrooms_with_furniture / number_of_livingrooms if number_of_livingrooms > 0 else 0
    
    number_of_floor_in_common_areas = len(
        [obj for obj in outputs 
            if obj.roomClass == 'common_area' 
            and obj.className in ['raw floor', 'intermediate loor', 'finished floor']
        ]
    )
    number_of_wall_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className in ['raw wall', 'started wall', 'finished wall']
        ]
    )
    number_of_ceiling_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className in ['raw ceiling', 'finished ceiling']
        ]
    )

    raw_floor_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'raw floor'
        ]
    )

    intermediate_floor_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'intermediate floor'
        ]
    )

    finished_floor_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'finished floor'
        ]
    )
    
    stats.commonAreasRawFloor = raw_floor_in_common_areas / number_of_floor_in_common_areas if number_of_floor_in_common_areas > 0 else 0
    stats.commonAreasIntermediateFloor = intermediate_floor_in_common_areas / number_of_floor_in_common_areas if number_of_floor_in_common_areas > 0 else 0
    stats.commonAreasFinishedFloor = finished_floor_in_common_areas / number_of_floor_in_common_areas if number_of_floor_in_common_areas > 0 else 0

    raw_wall_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'raw wall'
        ]
    )

    started_wall_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'started wall'
        ]
    )

    finished_wall_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'finished wall'
        ]
    )

    stats.commonAreasRawWall = raw_wall_in_common_areas / number_of_wall_in_common_areas if number_of_wall_in_common_areas > 0 else 0
    stats.commonAreasStartedWall = started_wall_in_common_areas / number_of_wall_in_common_areas if number_of_wall_in_common_areas > 0 else 0
    stats.commonAreasFinishedWall = finished_wall_in_common_areas / number_of_wall_in_common_areas if number_of_wall_in_common_areas > 0 else 0

    raw_ceiling_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'raw ceiling'
        ]
    )

    finished_ceiling_in_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass == 'common_area'
            and obj.className == 'finished ceiling'
        ]
    )

    stats.commonAreasRawCeiling = raw_ceiling_in_common_areas / number_of_ceiling_in_common_areas if number_of_ceiling_in_common_areas > 0 else 0
    stats.commonAreasFinishedCeiling = finished_ceiling_in_common_areas / number_of_ceiling_in_common_areas if number_of_ceiling_in_common_areas > 0 else 0

    number_of_walls_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className in ['raw wall', 'started wall', 'finished wall']
        ]
    )

    number_of_floors_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className in ['raw floor', 'intermediate floor', 'finished floor']
        ]
    )

    number_of_ceilings_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className in ['raw ceiling', 'finished ceiling']
        ]
    )

    raw_wall_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'raw wall'
        ]
    )

    started_wall_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'started wall'
        ]
    )

    finished_wall_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'finished wall'
        ]
    )

    stats.allButCommonRawWall = raw_wall_in_all_but_common_areas / number_of_walls_in_all_but_common_areas if number_of_walls_in_all_but_common_areas > 0 else 0
    stats.allButCommonStartedWall = started_wall_in_all_but_common_areas / number_of_walls_in_all_but_common_areas if number_of_walls_in_all_but_common_areas > 0 else 0
    stats.allButCommonFinishedWall = finished_wall_in_all_but_common_areas / number_of_walls_in_all_but_common_areas if number_of_walls_in_all_but_common_areas > 0 else 0

    raw_floor_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'raw floor'
        ]
    )

    intermediate_floor_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'intermediate floor'
        ]
    )

    finished_floor_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'finished floor'
        ]
    )

    stats.allButCommonRawFloor = raw_floor_in_all_but_common_areas / number_of_floors_in_all_but_common_areas if number_of_floors_in_all_but_common_areas > 0 else 0    
    stats.allButCommonIntermediateFloor = intermediate_floor_in_all_but_common_areas / number_of_floors_in_all_but_common_areas if number_of_floors_in_all_but_common_areas > 0 else 0
    stats.allButCommonFinishedFloor = finished_floor_in_all_but_common_areas / number_of_floors_in_all_but_common_areas if number_of_floors_in_all_but_common_areas > 0 else 0

    raw_ceiling_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'raw ceiling'
        ]
    )

    finished_ceiling_in_all_but_common_areas = len(
        [obj for obj in outputs
            if obj.roomClass != 'common_area'
            and obj.className == 'finished ceiling'
        ]
    )

    stats.allButCommonRawCeiling = raw_ceiling_in_all_but_common_areas / number_of_ceilings_in_all_but_common_areas if number_of_ceilings_in_all_but_common_areas > 0 else 0
    stats.allButCommonFinishedCeiling = finished_ceiling_in_all_but_common_areas / number_of_ceilings_in_all_but_common_areas if number_of_ceilings_in_all_but_common_areas > 0 else 0

    any_doorways = len(
        [obj for obj in outputs
            if (obj.className == 'no-door-doorway' or obj.className == 'door-way')
            and (obj.roomClass == 'livingroom' or obj.roomClass == 'kitchen')
        ]
    )

    doorways = len(
        [obj for obj in outputs
            if obj.className == 'door-way'
            and obj.roomClass == 'livingroom' or obj.roomClass == 'kitchen'
        ]
    )

    stats.allDoors = doorways / any_doorways if any_doorways > 0 else 0

    num_of_windows = len(
        [obj for obj in outputs
            if (obj.className == 'window' or obj.className == 'raw window')
            and obj.roomClass == 'livingroom' or obj.roomClass == 'kitchen'
        ]
    )

    num_of_finished_windowsills = len(
        [obj for obj in outputs
            if obj.className == 'finished windowsill'
            and obj.roomClass == 'livingroom' or obj.roomClass == 'kitchen'
        ]
    )

    stats.livingOrKitchenWindow = num_of_finished_windowsills / num_of_windows if num_of_windows > 0 else 0   

    final_score = score(stats)
    stats.final_score = float(final_score)

    return stats


def __score_non_common_area(stats: ScoreCardReport) -> float:
    # the score for walls in non common areas
    score_wall_non_common = int(stats.allButCommonFinishedWall + stats.allButCommonStartedWall + stats.allButCommonRawWall) == 0  
    +  (4 / 7) * stats.allButCommonFinishedWall + (2 / 7) * (1 - stats.allButCommonStartedWall) + (1 / 7) * (1 - stats.allButCommonRawWall)
    
    # the score for the floor in non common areas
    score_floor_non_common = int(stats.allButCommonFinishedFloor + stats.allButCommonIntermediateFloor + stats.allButCommonRawFloor) == 0  
    +  (4 / 7) * stats.allButCommonFinishedFloor + (2 / 7) * (1 - stats.allButCommonIntermediateFloor) + (1 / 7) * (1 - stats.allButCommonRawFloor)
    
    # the score for the ceiling in non commons areas
    score_ceiling_non_common = int(stats.allButCommonFinishedCeiling + stats.allButCommonRawCeiling == 0)  
    +  (2 / 3) * stats.allButCommonFinishedCeiling +  (1 / 3) * (1 -stats.allButCommonRawCeiling)

    # the score for the common area is the average for the values calculated above
    score_non_common = np.mean([score_wall_non_common, score_floor_non_common, score_ceiling_non_common])

    return float(score_non_common) # using np.mean return a numpy.float object


def __score_common_area(stats: ScoreCardReport) -> float:
    # the calculations are similar to those of the  common area
    # score for wall in common area
    common_score_wall = int(stats.commonAreasFinishedWall + stats.commonAreasStartedWall + stats.commonAreasRawWall) == 0  
    +  (4 / 7) * stats.commonAreasFinishedWall + (2 / 7) *  (1 - stats.commonAreasStartedWall) + (1 / 7) *  (1 - stats.commonAreasRawWall)
    
    #score for floor in common area
    common_score_floor = int(stats.commonAreasFinishedFloor + stats.commonAreasIntermediateFloor + stats.commonAreasRawFloor) == 0  
    +  (4 / 7) * stats.commonAreasFinishedFloor + (2 / 7) *  (1 - stats.commonAreasIntermediateFloor) + (1 / 7) *  (1 - stats.commonAreasRawFloor)
    
    # score for ceiling in common area
    common_score_ceiling = int(stats.commonAreasFinishedCeiling + stats.commonAreasRawCeiling) == 0  
    +  (2 / 3) * stats.commonAreasFinishedCeiling + (1 / 3) *  (1 - stats.commonAreasRawCeiling)

    # the score for the common area is the average for the values calculated above
    score_common = np.mean([common_score_wall, common_score_ceiling, common_score_floor])

    return float(score_common)


def __score_bathroom(stats: ScoreCardReport) -> float:
    return float(np.mean([stats.bathroomToiletBathtub + stats.bathroomToiletSeat + stats.bathroomToiletSink]))

def __score_kitchen(stats: ScoreCardReport) -> float:
    return float(np.mean([stats.livingOrKitchenFurniture, stats.livingOrKitchenRadiator, stats.livingOrKitchenWindow]))


DEFAULT_WEIGHTS = [0.5, 0.25, 0.125, 0.125]

def score(stats: ScoreCardReport, weights: Sequence = None, score_for_socket: float= 0.01, penalty_for_garbage: float = -0.02):

    # set the weights to their default values if none were given
    if weights is None:
        weights = np.array(DEFAULT_WEIGHTS)
    
    intermediate_scores = np.array([[__score_non_common_area(stats)], [__score_common_area(stats)], [__score_kitchen(stats)], [__score_bathroom(stats)]])

    # the initial score without considering sockets or garbage    
    score = weights @ intermediate_scores

    score += score_for_socket * stats.allSocketsAndSwitches + penalty_for_garbage * stats.allGarbage
    
    return min(1, max(score, 0))

