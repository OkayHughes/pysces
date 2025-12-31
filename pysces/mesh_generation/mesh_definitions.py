"""
This codebase uses the following convention for edge orientations
for 2D elements.
       E1
    [v1 → v2]
 E2 [↓    ↓] E3
    [v3 → v4]
       E4
We use the cubed-sphere convention
              --------------
              |            |
              |            |
              |     Top    |
              |            |
              |            |
  ----------------------------------------------------
  |           |            |            |            |
  |           |            |            |            |
  |   Left    |    Front   |    Right   |    Back    |
  |           |↑           |            |            |
  |           |·→          |            |            |
  ----------------------------------------------------
              |            |
              |            |
              |   Bottom   |
              |            |
              |            |
              --------------
where the above diagram is folded along non-conforming edges
to form a cube, and cartesian dimensions in each cubed-sphere face
are derived from the ambient cartesian coordinates in the flattened form.
"""
BACKWARDS, FORWARDS = (0, 1)
TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE = (0, 1, 2, 3, 4, 5)
TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE = (0, 1, 2, 3)
