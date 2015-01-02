% ---------------------------------------------------------------------
% EG author guidelines plus sample file for EG publication using LaTeX2e input
% D.Fellner, v1.16, Jan 21, 2009


\title[Exploring Ray-Space Hierarchy on the GPU for Ray-Tracing of Dynamic Scenes]%
      {Exploring Ray-Space Hierarchy on the GPU for Ray-Tracing of Dynamic Scenes}

\author[Nuno Tiago Reis]
       {
	Nuno Tiago Reis\\
	Instituto Superior Técnico, Portugal\\
       }

% ------------------------------------------------------------------------

% if the Editors-in-Chief have given you the data, you may uncomment
% the following five lines and insert it here
%
% \volume{27}   % the volume in which the issue will be published;
% \issue{1}     % the issue number of the publication
% \pStartPage{1}      % set starting page


%-------------------------------------------------------------------------
\begin{document}

\maketitle

\begin{abstract}

	Overview of the work i produced.

%\begin{classification} % according to http://www.acm.org/class/1998/
%\CCScat{Computer Graphics}{I.3.3}{Picture/Image Generation}{Line and curve generation}
%\end{classification}

\end{abstract}

%-------------------------------------------------------------------------
\section{Introduction}

	Ray-tracing is a global illumination technique used for the synthesis of photo-realistic images that involves ray-casting.
	The first ray-casting algorithm was presented by Arthur Appel \cite{Appel68}, although it was only referred to as such 
	until later on. This algorithm involved casting a set number of rays from the eye and intersecting those rays with 
	the geometry composing the scene. 

	In 1979 Whitted \cite{Whitted80} presented a paper introducing a recursive ray-tracing 
	algorithm, which is now known as Whitted Ray-Tracing. This algorithm casts rays, much like the original algorithm,
	casts rays from the eye but doesn't stop there. If the rays intersect geometry they can generate three different kinds
	of rays: shadow, reflection and refraction rays. These rays are also called secondary rays.

	Shadow rays are cast towards the light sources present in the scene. These rays are cast to check if the intersected object is
	being illuminated directly or if there are other objects blocking the light source. 
	Reflection rays are cast in a mirror-reflection direction. However, these rays are only cast if the intersected object has 
	reflective properties. 
	Refraction rays are cast through objects. This means that the intersected object must be translucent and that the refraction
	direction also depends on the material of both the rays current and next travel mediums refraction index.

	These rays differentiate ray-tracing from rasterization since they allow realistic and complex effects like reflections,
	refractions and shadows. This doesn't come without a price however. Ray-tracing is very inefficient and up to this day there
	has been extensive research to try and optimize it. Most of this research involves creating hierarchies. 

	In Whitted Ray-Tracing \cite{Whitted80} each ray is tested with each polygon in the scene, which leads to N x M intersection tests
	per frame, assuming N rays and M polygons. This isn't feasible, especially with moderately complex scenes. 

	Object-space hierarchies help reduce the number of intersections by creating a structure that allows the scenes geometry to be 
	tested for intersections without actually testing every single polygon. These techniques have problems handling dynamic scenes 
	since the hierarchies must be rebuilt at each frame.

	Ray-space hierarchies have the same goal, to reduce the number of intersections, but they approach the problem differently. 
	One of the solutions is ray-bundling, which consists in grouping several coherent rays together and intersect the bundle of rays with
	the geometry. This saves many intersection tests since if a bundle doesn't intersect any geometry then the rays in said bundle
	will not be tested anymore in that particular frame. Another solution involve caching of rays, using 5D tuples to represent them. 
	In these solutions the rays are represented as hypercubes in 5D space, which are then intersected with the scenes geometry. 
	These intersection results are then cached and used to reduce future intersection tests for new rays.

	Over the last few years many improvements have been made in both object-space and ray-space approaches. The development of 
	General-purpose computing on Graphic Processing Units (GPGPU) over the past years has led to many attempts to implement 
	Ray-tracing on the GPU. This is no simple task however. Modern GPUs are composed of Single instruction, Multiple data units (SIMD),
	which means that they are optimized to repeat the same operations over large sets of input. Ray-tracing by nature isn't
	easily mapped onto GPUs since each ray can generate secondary rays, but it also might not do so, which is a problem since
	each ray-cast has the potential to have a different execution than the last one.

	In this paper I will analyse two different approaches based on ray-space hierarchies, one using caching mechanisms and 
	one using ray bundles. 
	Finally I will propose an improvement to a ray-bundle approach, combining both ray-space and object-space hierarchies to
	achieve higher efficiency in the ray-object intersection tests. This ray-bundling approach will also be parallelized so
	that it is mapped more efficiently on the GPU, leading to better performance, with the goal of generating photorealistic images in real time.

%-------------------------------------------------------------------------
\section{Related Work}

	\subsection{Ray-Space Hierarchies}

	\cite{ArvoKirk87}

	\cite{GaranzhaLoop10}

	\cite{Roger07}

	\cite{SimiakisDay94}

	\subsection{Object -Space Hierarchies}



	\subsection{Ray-Tracing on the GPU}

	\cite{Roger07}

%-------------------------------------------------------------------------
\section{Original Algorithm}

	TODO

\subsection{Overview}

	TODO

%-------------------------------------------------------------------------
\subsection{Improved Algorithm}

	TODO

\subsection{Overview}

	TODO

%------------------------------------------------------------------------
\section{Analysis}

	TODO
	
%-------------------------------------------------------------------------
\section{Results}

	TODO

%%%
%%% Figure 1
%%%
\begin{figure}[htb]
  \centering
  % the following command controls the width of the embedded PS file
  % (relative to the width of the current column)
  \includegraphics[width=.8\linewidth]{figure-1}
  % replacing the above command with the one below will explicitly set
  % the bounding box of the PS figure to the rectangle (xl,yl),(xh,yh).
  % It will also prevent LaTeX from reading the PS file to determine
  % the bounding box (i.e., it will speed up the compilation process)
  % \includegraphics[width=.95\linewidth, bb=39 696 126 756]{figure-1}
  %
  \parbox[t]{.9\columnwidth}{\relax
           For all figures please keep in mind that you \textbf{must not}
           use images with transparent background! 
           }
  %
  \caption{\label{fig:firstExample}
           Here is a sample figure.}
\end{figure}

%All graphics should be centered.

%If your paper includes images, it is very important that they are of
%sufficient resolution to be faithfully reproduced.

%To determine the optimum size (width and height) of an image, measure
%the image's size as it appears in your document (in millimeters), and
%then multiply those two values by 12. The resulting values are the
%optimum $x$ and $y$ resolution, in pixels, of the image. Image quality
%will suffer if these guidelines are not followed.

%Example 1: 
%
%An image measures 50\,mm by 75\,mm when placed in a document. This
%image should have a resolution of no less than 600 pixels by 900
%pixels in order to be reproduced faithfully.

%Example 2: 
%
%Capturing a screenshot of your entire $1024 \times 768$ pixel display
%monitor may be useful in illustrating a concept from your research. In
%order to be reproduced faithfully, that $1024 \times 768$ image should
%be no larger than 85 mm by 64 mm (approximately) when placed in your
%document.

%-------------------------------------------------------------------------

\bibliographystyle{eg-alpha}

\bibliography{references}

\end{document}
