## readme.txt 内容

```
Project 4: Calibration and Augmented Reality
Team: Junyao Han, Junrui Ding

Time travel days used: 0

------------------------------------------------------------
Building the project
------------------------------------------------------------
cd cmake-build-debug
cmake ..
make

This produces three executables:
  project4          – Tasks 1-3 (calibration)
  project4_pose     – Tasks 4-6 + Extensions 1-2
  project4_features – Task 7 (Harris corner detection)

------------------------------------------------------------
Running the programs
------------------------------------------------------------

Step 1 – Calibration (Tasks 1-3)
  ./project4
  Hold a 9x6 checkerboard in front of the camera.
  Press 's' to save a calibration frame (need at least 5).
  Press 'c' to run calibration.
  Press 'w' to write intrinsics.yml.
  Press 'q' to quit.

Step 2 – AR with church (Tasks 4-6, Extensions 1-2)
  Live mode:
    ./project4_pose
    Show a 9x6 checkerboard (Board A) to see the church.
    Optionally show a 5x4 checkerboard (Board B) to see the arrow sign.
    Press 's' to save a screenshot, 'q' to quit.

  Static image mode (Extension 2):
    ./project4_pose image_path

Step 3 – Harris corner detection (Task 7)
  ./project4_features
  Point the camera at any scene.
  Use the Threshold and Block Size trackbars to tune detection.
  Press '+'/'-' to adjust the Harris k parameter.
  Press 's' to save a screenshot, 'q' to quit.

------------------------------------------------------------
Extensions implemented
------------------------------------------------------------
1. Two targets: Board A (9x6) displays a 3D church. Board B (5x4)
   displays an arrow sign that dynamically rotates to point toward
   the church based on the tvec difference between the two boards.

2. Static image mode: pass any image path as a command-line argument
   to project4_pose to run AR on a pre-captured photo.
   Note: Static image mode only supports one target and displays a 3D church based on our design.
```
