# Object Recognition via Image Features Detection & Matching in OpenCV-Python

  <img src="images/asift_Line_test_a_test_b.png" width="1000"/>
  

## 1.  Objectives

The objective of this section is to demonstrate feature matching and object detection using OpenCV Python. 

## 2.  Object recognition via feature matching

* A computer vision object recognition process has the following steps:

  * Acquire the image of the scene
  * Detect key points and features
  * Try to match detected features with those extracted from known objects stored in the data base
  * Identify the new imaged object or scene based on any successful matches in step 3. 
  * That is the object is associated with the data base object yielding a sufficiently high number of matches, if there is one.


In this section, we shall implement and illustrate a selected set of image feature matching and object recognition algorithms that are available in OpenCV. In particular, we shall illustrate how to match features in one image with another using the following feature matching algorithms in OpenCV:

  * Brute-Force matcher
  * FLANN Matcher.


Next, we present the input query and scene images.


## 3.  Input Images

The query and scene images are illustrated in the figure below. Our objective is to detect and localize the query image in the scene image.

<img src="images/input-images - 02.jpg" width="1000"/>

## 4.  Development

* Author: Mohsen Ghazel (mghazel)
* Date: March 31st, 2021
* Project: Image Features Matching & Object Recognition:
  * The objective of this project is to demonstrate how to match features and recognize objects in a scene using OpenCV with Python API
  * We shall implement various types of features matching algorithms.
  
### 4.1. Step 1: Python Imports:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># system environment</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># Numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># image processing library</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>image <span style="color:#800000; font-weight:bold; ">as</span> mpimg
<span style="color:#696969; "># date and time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime

<span style="color:#696969; "># import warnings</span>
<span style="color:#800000; font-weight:bold; ">import</span> warnings
<span style="color:#696969; "># suppress warnings</span>
warnings<span style="color:#808030; ">.</span>filterwarnings<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"ignore"</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Feb <span style="color:#008c00; ">20</span> <span style="color:#008c00; ">2021</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">17</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">23</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span> 


<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#074726; ">__doc__</span><span style="color:#808030; ">)</span>
Automatically created module <span style="color:#800000; font-weight:bold; ">for</span> IPython interactive environment 

</pre>

### 4.2. Step 2: Read and visualize the input images:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.1) Read the query-image</span>
<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># The input query image file name</span>
query_img_file_path_name <span style="color:#808030; ">=</span> os<span style="color:#808030; ">.</span>path<span style="color:#808030; ">.</span>join<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"sample_data"</span><span style="color:#808030; ">,</span><span style="color:#0000e6; ">"box.png"</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># read the input query-image</span>
img_query <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>imread<span style="color:#808030; ">(</span>query_img_file_path_name<span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if the query-image is read successfully</span>
<span style="color:#800000; font-weight:bold; ">if</span> img_query <span style="color:#800000; font-weight:bold; ">is</span> <span style="color:#074726; ">None</span><span style="color:#808030; ">:</span>
    sys<span style="color:#808030; ">.</span>exit<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Could not read the image file: "</span> <span style="color:#44aadd; ">+</span> query_img_file_path_name<span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if it is grayscale image, if so convert it to RGB by </span>
<span style="color:#696969; "># duplicating the channel</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  mg_query <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>merge<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>mg_query<span style="color:#808030; ">,</span>mg_query<span style="color:#808030; ">,</span>mg_query<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if it is color image, if so convert it to grayscale</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    gray_query <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>cvtColor<span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>COLOR_BGR2GRAY<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># make a copy of the query-image</span>
    gray_query <span style="color:#808030; ">=</span> img_query<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.2) Read the scene image</span>
<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># The input scene image file name</span>
scene_img_file_path_name <span style="color:#808030; ">=</span> os<span style="color:#808030; ">.</span>path<span style="color:#808030; ">.</span>join<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"sample_data"</span><span style="color:#808030; ">,</span><span style="color:#0000e6; ">"box-in-scene.png"</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># read the input scene-image</span>
img_scene <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>imread<span style="color:#808030; ">(</span>scene_img_file_path_name<span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if the scene-image is read successfully</span>
<span style="color:#800000; font-weight:bold; ">if</span> img_scene <span style="color:#800000; font-weight:bold; ">is</span> <span style="color:#074726; ">None</span><span style="color:#808030; ">:</span>
    sys<span style="color:#808030; ">.</span>exit<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Could not read the image file: "</span> <span style="color:#44aadd; ">+</span> scene_img_file_path_name<span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if it is grayscale image, if so convert it to RGB by </span>
<span style="color:#696969; "># duplicating the channel</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img_scene<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  mg_scene <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>merge<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>mg_scene<span style="color:#808030; ">,</span>mg_scene<span style="color:#808030; ">,</span>mg_scene<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if it is color image, if so convert it to grayscale</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img_scene<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    gray_scene <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>cvtColor<span style="color:#808030; ">(</span>img_scene<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>COLOR_BGR2GRAY<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># make a copy of the scene-image</span>
    gray_scene <span style="color:#808030; ">=</span> img_scene<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.3) Display the scene and the query images:</span>
<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># create a figure</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Input scene and query images"</span><span style="color:#808030; ">,</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.3.1) display the input query-image </span>
<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># display the original query-image</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">122</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Query-image"</span><span style="color:#808030; ">,</span> fontsize <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">122</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the query-image</span>
<span style="color:#696969; "># - if the image is RGB</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># for grayscale image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># 2.3.2) display the input scene-image </span>
<span style="color:#696969; ">#-----------------------------------------------------------------------</span>
<span style="color:#696969; "># create a figure</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Input scene and query images"</span><span style="color:#808030; ">,</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the original scene-image</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">121</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Scene-image"</span><span style="color:#808030; ">,</span> fontsize <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the scene image</span>
<span style="color:#696969; "># - if the image is RGB</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img_scene<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img_scene<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># for grayscale image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img_scene<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/input-images.jpg" width="1000"/>

### 4.3. Step 3: Brute-Force Feature Matching:

* Brute-Force matcher is simple:
  * It takes the descriptor of each query-image feature in first and matches it with all other features in scene-image using some distance calculation and the closest one is returned.
  * This process is repeated for all the features
  * In the end we pick the K features, based on the distances separating the query feature and its matched reference-image feature.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#=================================================================</span>
<span style="color:#696969; "># 3) Brute-Force Feature Matcher</span>
<span style="color:#696969; ">#=================================================================</span>
<span style="color:#696969; "># Step 1: Detect the features from the scene and query </span>
<span style="color:#696969; ">#         images:</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; ">#         - We compute the ORB features</span>
<span style="color:#696969; ">#         - One may experiment with computing other features</span>
<span style="color:#696969; ">#           such as Harris corners, FAST, BRIEF, etc.</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># Initiate ORB detector</span>
orb <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>ORB_create<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># find the keypoints and descriptors with ORB</span>
<span style="color:#696969; "># - for the query image</span>
kp_query<span style="color:#808030; ">,</span> des_query <span style="color:#808030; ">=</span> orb<span style="color:#808030; ">.</span>detectAndCompute<span style="color:#808030; ">(</span>gray_query<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># - for the scene image</span>
kp_scene<span style="color:#808030; ">,</span> des_scene <span style="color:#808030; ">=</span> orb<span style="color:#808030; ">.</span>detectAndCompute<span style="color:#808030; ">(</span>gray_scene<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># Step 2: Match the scene and query images features using the </span>
<span style="color:#696969; ">#         Brute-Force Matcher.</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># create BFMatcher object</span>
bf <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>BFMatcher<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>NORM_HAMMING<span style="color:#808030; ">,</span> crossCheck<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># Match descriptors.</span>
matches <span style="color:#808030; ">=</span> bf<span style="color:#808030; ">.</span>match<span style="color:#808030; ">(</span>des_query<span style="color:#808030; ">,</span>des_scene<span style="color:#808030; ">)</span>

<span style="color:#696969; "># Sort them in the order of their distance.</span>
matches <span style="color:#808030; ">=</span> <span style="color:#400000; ">sorted</span><span style="color:#808030; ">(</span>matches<span style="color:#808030; ">,</span> key <span style="color:#808030; ">=</span> <span style="color:#800000; font-weight:bold; ">lambda</span> x<span style="color:#808030; ">:</span>x<span style="color:#808030; ">.</span>distance<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># Step 3: Visualize the matches</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># visualization preferences parameters</span>
draw_params <span style="color:#808030; ">=</span> <span style="color:#400000; ">dict</span><span style="color:#808030; ">(</span>matchColor <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>  <span style="color:#696969; "># matching-lines color</span>
                   singlePointColor <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#696969; ">#  keypoints color</span>
                   flags <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>DrawMatchesFlags_DEFAULT<span style="color:#808030; ">)</span> <span style="color:#696969; "># show kepoints and matching lines</span>
<span style="color:#696969; "># Draw first 15 matches.</span>
img3 <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>drawMatches<span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">,</span>kp_query<span style="color:#808030; ">,</span>img_scene<span style="color:#808030; ">,</span>kp_scene<span style="color:#808030; ">,</span>matches<span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">15</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span><span style="color:#44aadd; ">**</span>draw_params<span style="color:#808030; ">)</span>
<span style="color:#696969; "># create the figure</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"BFMatcher - Best Match"</span><span style="color:#808030; ">,</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">111</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Brute-Force Matcher: The 15 matches"</span><span style="color:#808030; ">,</span> fontsize <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img3<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/brute-force-matches.jpg" width="1000"/>

### 4.4. Step 4: FLANN Based Feature Matching:

* FLANN stands for Fast Library for Approximate Nearest Neighbors:
  * It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features
  * It works faster than BFMatcher for large datasets.
  
 
<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># 4) FLANN-Based Feature Matcher</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># 4.1) Detect the features from the scene and query </span>
<span style="color:#696969; ">#      images:</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; ">#         - We compute the ORB features</span>
<span style="color:#696969; ">#         - One may experiment with computing other features</span>
<span style="color:#696969; ">#           such as Harris corners, FAST, BRIEF, etc.</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># Initiate ORB detector</span>
orb <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>ORB_create<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># find the keypoints and descriptors with ORB</span>
<span style="color:#696969; "># - for the query image</span>
kp_query<span style="color:#808030; ">,</span> des_query <span style="color:#808030; ">=</span> orb<span style="color:#808030; ">.</span>detectAndCompute<span style="color:#808030; ">(</span>gray_query<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># - for the scene image</span>
kp_scene<span style="color:#808030; ">,</span> des_scene <span style="color:#808030; ">=</span> orb<span style="color:#808030; ">.</span>detectAndCompute<span style="color:#808030; ">(</span>gray_scene<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># 4.2) Apply FLANN matcher</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># When using ORB features, you can pass the following. </span>
FLANN_INDEX_LSH <span style="color:#808030; ">=</span> <span style="color:#008c00; ">6</span>
index_params<span style="color:#808030; ">=</span> <span style="color:#400000; ">dict</span><span style="color:#808030; ">(</span>algorithm <span style="color:#808030; ">=</span> FLANN_INDEX_LSH<span style="color:#808030; ">,</span>
               table_number <span style="color:#808030; ">=</span> <span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span> <span style="color:#696969; "># 12</span>
               key_size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span>     <span style="color:#696969; "># 20</span>
               multi_probe_level <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span> <span style="color:#696969; ">#2</span>
<span style="color:#696969; "># search params</span>
search_params <span style="color:#808030; ">=</span> <span style="color:#400000; ">dict</span><span style="color:#808030; ">(</span>checks<span style="color:#808030; ">=</span><span style="color:#008c00; ">50</span><span style="color:#808030; ">)</span>   <span style="color:#696969; "># or pass empty dictionary</span>
<span style="color:#696969; "># create the FLANN matcher</span>
flann <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>FlannBasedMatcher<span style="color:#808030; ">(</span>index_params<span style="color:#808030; ">,</span> search_params<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#  We apply the knnMatch() to get k best matches. </span>
<span style="color:#696969; "># - In this example, we will take k=2 so that we can apply ratio test</span>
matches <span style="color:#808030; ">=</span> flann<span style="color:#808030; ">.</span>knnMatch<span style="color:#808030; ">(</span>des_query<span style="color:#808030; ">,</span> des_scene<span style="color:#808030; ">,</span> k<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># Need to draw only good matches, so create a mask</span>
matchesMask <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>matches<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">]</span>

<span style="color:#696969; "># ratio test as per Lowe's paper</span>
<span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span><span style="color:#808030; ">(</span>m<span style="color:#808030; ">,</span>n<span style="color:#808030; ">)</span> <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">enumerate</span><span style="color:#808030; ">(</span>matches<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#800000; font-weight:bold; ">if</span> m<span style="color:#808030; ">.</span>distance <span style="color:#44aadd; ">&lt;</span> <span style="color:#008000; ">0.75</span><span style="color:#44aadd; ">*</span>n<span style="color:#808030; ">.</span>distance<span style="color:#808030; ">:</span>
        matchesMask<span style="color:#808030; ">[</span>i<span style="color:#808030; ">]</span><span style="color:#808030; ">=</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>

<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># 4.3) Visualize the matches</span>
<span style="color:#696969; ">#-----------------------------------------------------------------  </span>
<span style="color:#696969; "># visualization preferences parameters       </span>
draw_params <span style="color:#808030; ">=</span> <span style="color:#400000; ">dict</span><span style="color:#808030; ">(</span>matchColor <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>  <span style="color:#696969; "># matching-lines color</span>
                   singlePointColor <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#696969; ">#  keypoints color</span>
                   matchesMask <span style="color:#808030; ">=</span> matchesMask<span style="color:#808030; ">,</span>
                   flags <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>DrawMatchesFlags_DEFAULT<span style="color:#808030; ">)</span> <span style="color:#696969; "># show kepoints and matching lines</span>
<span style="color:#696969; "># overlay the matches on the images</span>
img3 <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>drawMatchesKnn<span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">,</span>kp_query<span style="color:#808030; ">,</span>img_scene<span style="color:#808030; ">,</span>kp_scene<span style="color:#808030; ">,</span>matches<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span><span style="color:#44aadd; ">**</span>draw_params<span style="color:#808030; ">)</span>

<span style="color:#696969; "># create the figure</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"FLANN-Based Feature Matching"</span><span style="color:#808030; ">,</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">111</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"FLANN-Based Feature Matching"</span><span style="color:#808030; ">,</span> fontsize <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img3<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/FLANN-matches.jpg" width="1000"/>

### 4.5. Step 5: Query Image Recognition:

* The matched features are used to estimate the Homography matrix:

  * The Homography matrix H is a 3x3 matrix, which provides a linear transformation between query and scene images
  * It transforms the query-image plane P1 to the scene-image plane P2.
  * Once the Homography matrix H is estimated:
  * We can then localize the query image in the scene image by mapping its four corners/vertices using the Homography matrix H.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-----------------------------------------------------------------  </span>
<span style="color:#696969; "># Step 5: Query Image Recognition: FLANN-Based Feature Matching</span>
<span style="color:#696969; ">#-----------------------------------------------------------------  </span>
<span style="color:#696969; "># 5.1) Store all the good matches satisfying the ratio test.</span>
<span style="color:#696969; ">#-----------------------------------------------------------------  </span>
good <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">]</span>
<span style="color:#696969; "># filter all the matches:</span>
<span style="color:#696969; ">#  - only keep the good matches based on Ratio Test</span>
<span style="color:#800000; font-weight:bold; ">for</span> m<span style="color:#808030; ">,</span>n <span style="color:#800000; font-weight:bold; ">in</span> matches<span style="color:#808030; ">:</span>
    <span style="color:#800000; font-weight:bold; ">if</span> m<span style="color:#808030; ">.</span>distance <span style="color:#44aadd; ">&lt;</span> <span style="color:#008000; ">0.75</span><span style="color:#44aadd; ">*</span>n<span style="color:#808030; ">.</span>distance<span style="color:#808030; ">:</span>
        good<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span>m<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># 5.2) Compute the Homography matrix and map the query image to </span>
<span style="color:#696969; ">#      the scene image</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - This can only be done if we have a sufficient number of matches:</span>
<span style="color:#696969; "># - The minimum number of good matches required </span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
MIN_MATCH_COUNT <span style="color:#808030; ">=</span> <span style="color:#008c00; ">10</span>
<span style="color:#696969; "># if we have sufficient number of good matches</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>good<span style="color:#808030; ">)</span><span style="color:#44aadd; ">&gt;</span>MIN_MATCH_COUNT<span style="color:#808030; ">:</span>
    <span style="color:#696969; "># the query image matches</span>
    src_pts <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>float32<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span> kp_query<span style="color:#808030; ">[</span>m<span style="color:#808030; ">.</span>queryIdx<span style="color:#808030; ">]</span><span style="color:#808030; ">.</span>pt <span style="color:#800000; font-weight:bold; ">for</span> m <span style="color:#800000; font-weight:bold; ">in</span> good <span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>reshape<span style="color:#808030; ">(</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># the scene image matches</span>
    dst_pts <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>float32<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span> kp_scene<span style="color:#808030; ">[</span>m<span style="color:#808030; ">.</span>trainIdx<span style="color:#808030; ">]</span><span style="color:#808030; ">.</span>pt <span style="color:#800000; font-weight:bold; ">for</span> m <span style="color:#800000; font-weight:bold; ">in</span> good <span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>reshape<span style="color:#808030; ">(</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># compute the homography by solving a system of equations</span>
    M<span style="color:#808030; ">,</span> mask <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>findHomography<span style="color:#808030; ">(</span>src_pts<span style="color:#808030; ">,</span> dst_pts<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>RANSAC<span style="color:#808030; ">,</span><span style="color:#008000; ">5.0</span><span style="color:#808030; ">)</span>
  <span style="color:#696969; "># display the homography matrix</span>
    <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The estimated Homography matrix: H = "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>M<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># process the mask to convert it list</span>
    matchesMask <span style="color:#808030; ">=</span> mask<span style="color:#808030; ">.</span>ravel<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>tolist<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># the shape of the query image</span>
    h<span style="color:#808030; ">,</span>w<span style="color:#808030; ">,</span>d <span style="color:#808030; ">=</span> img_query<span style="color:#808030; ">.</span>shape
    <span style="color:#696969; "># get the four corner so fthe query images</span>
    pts <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>float32<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span> <span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>h<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span><span style="color:#808030; ">[</span>w<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span>h<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span><span style="color:#808030; ">[</span>w<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>reshape<span style="color:#808030; ">(</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># map the four corners o fthe queery image using the estimated homography</span>
    dst <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>perspectiveTransform<span style="color:#808030; ">(</span>pts<span style="color:#808030; ">,</span>M<span style="color:#808030; ">)</span>
    <span style="color:#696969; "># overlay the boundaries of the detected query image on the scene image</span>
    img_scene <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>polylines<span style="color:#808030; ">(</span>img_scene<span style="color:#808030; ">,</span><span style="color:#808030; ">[</span>np<span style="color:#808030; ">.</span>int32<span style="color:#808030; ">(</span>dst<span style="color:#808030; ">)</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span><span style="color:#074726; ">True</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>LINE_AA<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># in case there is not enough good matches</span>
    <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Not enough matches are found - {0}/{1}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>good<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>MIN_MATCH_COUNT<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    matchesMask <span style="color:#808030; ">=</span> <span style="color:#074726; ">None</span>
    
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># 5.3) Visualize the matches</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># visualization preferences parameters</span>
draw_params <span style="color:#808030; ">=</span> <span style="color:#400000; ">dict</span><span style="color:#808030; ">(</span>matchColor <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>  <span style="color:#696969; "># matching-lines color</span>
                   singlePointColor <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#696969; ">#  keypoints color</span>
                   matchesMask <span style="color:#808030; ">=</span> matchesMask<span style="color:#808030; ">,</span> <span style="color:#696969; "># draw only inliers</span>
                   flags <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>DrawMatchesFlags_DEFAULT<span style="color:#808030; ">)</span> <span style="color:#696969; "># show kepoints and matching lines</span>

<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># 5.4) Overlay the final results on the query and scene images:</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - detected features for the query and scene images</span>
<span style="color:#696969; "># - The mactched features between the 2 images</span>
<span style="color:#696969; "># - The detected location of the query image in the scene image</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
img3 <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>drawMatches<span style="color:#808030; ">(</span>img_query<span style="color:#808030; ">,</span>kp_query<span style="color:#808030; ">,</span>img_scene<span style="color:#808030; ">,</span>kp_scene<span style="color:#808030; ">,</span>good<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span><span style="color:#44aadd; ">**</span>draw_params<span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Final Query Image Recognition"</span><span style="color:#808030; ">,</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">111</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Final Query Image Recognition"</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img3<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>

The estimated Homography matrix<span style="color:#808030; ">:</span> 

H <span style="color:#808030; ">=</span> 	<span style="color:#808030; ">[</span><span style="color:#808030; ">[</span> <span style="color:#008000; ">4.43847781e-01</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">1.17603871e-01</span>  <span style="color:#008000; ">1.17431301e+02</span><span style="color:#808030; ">]</span>
 	<span style="color:#808030; ">[</span><span style="color:#44aadd; ">-</span><span style="color:#008000; ">4.31125831e-03</span>  <span style="color:#008000; ">4.81667451e-01</span>  <span style="color:#008000; ">1.59213335e+02</span><span style="color:#808030; ">]</span>
 	<span style="color:#808030; ">[</span><span style="color:#44aadd; ">-</span><span style="color:#008000; ">3.03207257e-04</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">6.99782680e-05</span>  <span style="color:#008000; ">1.00000000e+00</span><span style="color:#808030; ">]</span><span style="color:#808030; ">]</span> 
</pre>

<img src="images/final-image-recognition-results.jpg" width="1000"/>

### 4.6. Step 6: End of Execution:

* Display a successful end of execution message


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">## display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">01</span> <span style="color:#008c00; ">00</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">51</span><span style="color:#808030; ">:</span><span style="color:#008000; ">48.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye! 
</pre>


## 5. Analysis

* In view of the illustrated results, we make the following observations:

  % The two feature matching methods yield sufficient number of correctly matches features between the query and scene images
  * The query image is perfectly localized in the scene image.


## 5. Future Work

* We propose to explore the following tasks:

  * To explore different types of feature detection and matching combinations for object detection and localization
  * To explore the effects of the following factors on objection recognition performance:
    * Occlusion
    * Illumination
    * Orientation
    * Scale
    * Presence of similar objects (ex. bottle of Regular Coke vs Diet Coke)


## 6. Reference

1. OpenCV. Feature Matching. https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
2. OpenCV. Feature Matching. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#
3. Kaggle. Object recognition using feature matching. https://www.kaggle.com/dataenergy/object-recognition-using-feature-matching
4. OpenCV. Feature Matching + Homography to find Objects. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html 
5. Strahinja Zivkovic. Feature Matching methods comparison in OpenCV. http://datahacker.rs/feature-matching-methods-comparison-in-opencv/ 
6. Feature Matching (Homography) Brute Force OpenCV Python Tutorial. https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/ 
7. OpenCV. Feature Matching with FLANN. https://www.ccoderun.ca/programming/doxygen/opencv/tutorial_feature_flann_matcher.html 
8. Aishwarya Singh  A Detailed Guide to the Powerful SIFT Technique for Image Matching (with Python code). https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/


