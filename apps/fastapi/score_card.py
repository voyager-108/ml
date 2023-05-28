from pydantic import BaseModel
from ..nn.yolov8.score import YOLOv8Objects, analyze_video

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




def derive_statistics(outputs: list[YOLOv8Objects]):
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

    return stats
