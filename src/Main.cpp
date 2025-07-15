#include <cmath>
#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include"shaderClass.h"
#include"VAO.h"
#include"VBO.h"
#include"EBO.h"
#include <glm/gtc/type_ptr.hpp>
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "squareCoordinates.h"
#include "Square.h"
#include "utils.cuh"
#include "updateTemp.h"


bool showGUI = true;
bool start = true;
int useCPU = true;
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_C && action == GLFW_PRESS)
		{showGUI = !showGUI; std::cout << "conf\n";}
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		{glfwSetWindowShouldClose(window, true); std::cout << "closing\n";}
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{start = !start; std::cout << "start/stop\n";}
}


int main()
{
	// number of squares in width (x)
	int W = 128;
	// number of squares in height (y)
	int H = W;
	// total number of squares
	int N = H * W;

	Square squares[N];
	Square squares_out[N];

	// absolute width
	int Width = 1.0f;
	// absolute height
	int Height = 1.0f;

	// step x axis
	float dx = Width/(float)W;
	// step y axis
	float dy = Height/(float)H;
	// diffussion constant
	float a = 0.3;

	// disk radius and its square
	float R = 0.15;
	float radius2 = R * R;

	// inside and outside temperatures
	float Tin = 90; float Tout = 0;

	// GUI
	int windowWidth = 900;
	int windowHeight = 900;
	float dt_slider_value = .5f;

	vec<2, int> center_coordinates = vec<2, int>(0, 0);
	initDisk(squares, W, H, dx, dy, Height, Width, radius2, Tin, Tout, &center_coordinates);

	// Initialize GLFW
	glfwInit();

	// Tell GLFW what version of OpenGL we are using 
	// In this case we are using OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Tell GLFW we are using the CORE profile
	// So that means we only have the modern functions
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a GLFWwindow object of 800 by 800 pixels, naming it "YoutubeOpenGL"
	GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Heat Transfer Simulation", nullptr, nullptr);
	// Error check if the window fails to create
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	// Introduce the window into the current context
	glfwMakeContextCurrent(window);

	//Load GLAD so it configures OpenGL
	gladLoadGL();
	// Specify the viewport of OpenGL in the Window
	// In this case the viewport goes from x = 0, y = 0, to x = 800, y = 800
	glViewport(0, 0, windowWidth, windowHeight);

	// Generates Shader object using shaders defualt.vert and default.frag
	Shader shaderProgram("default.vert", "default.frag");

	// Generates Vertex Array Object and binds it
	VAO VAO1;
	VAO1.Bind();

	// Generates Vertex Buffer Object and links it to vertices
	VBO VBO1(vertices, sizeof(vertices));
	// Generates Element Buffer Object and links it to indices
	EBO EBO1(indices, sizeof(indices));

	// Links VBO attributes such as coordinates and colors to VAO
	VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 6 * sizeof(float), nullptr);
	VAO1.LinkAttrib(VBO1, 1, 3, GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	// Unbind all to prevent accidentally modifying them
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

	// Gets ID of uniform called "scale"
	GLuint uniID = glGetUniformLocation(shaderProgram.ID, "scale");

	//
	mat4 view = mat4(1.0f);
	// mat4 projection = mat4(1.0f);
	mat4 projection = ortho(-1.0f, .0f, -1.0f, .0f);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
	ImGui_ImplOpenGL3_Init();

	double previousTime = glfwGetTime();
	int frameCount = 0;
	std::string textFPS;

	// Main while loop
	while (!glfwWindowShouldClose(window))
	{
		double currentTime = glfwGetTime();
		frameCount++;
		// Specify the color of the background
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		// Clean the back buffer and assign the new color to it
		glClear(GL_COLOR_BUFFER_BIT);
		// Take care of all GLFW events
		glfwPollEvents();
		glfwSetKeyCallback(window, key_callback);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		//using namespace ImGui;
		ImGui::BeginMainMenuBar();
		float centerTemp = squares[getIndex(center_coordinates.x, center_coordinates.y, H)].temp;
		char displayText[64];
		sprintf(displayText, "Center Temperature: %.1f", centerTemp);
		ImGui::Text(displayText);
		char displayText2[64];
		ImGui::Separator();
		sprintf(displayText2, "Total Elements: %d", N);
		ImGui::Text(displayText2);

		char displayText3[64];
		float elapsedTime = glfwGetTime();
		sprintf(displayText3, "Time Elapsed: %.2f s", elapsedTime);
		ImGui::Separator();
		ImGui::Text(displayText3);

		if ( currentTime - previousTime >= 1.0 )
		{
			// Display the frame count here any way you want.
			textFPS = std::to_string(frameCount);

			frameCount = 0;
			previousTime = currentTime;
		}
		ImGui::Separator();
		ImGui::Text("Framerate: ");
		ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), (textFPS + " FPS").c_str());
		ImGui::EndMainMenuBar();

		if (showGUI) {
			ImGui::Begin("Configuration");
			ImGui::Text("Simulation Engine");
			ImGui::RadioButton("CPU", &useCPU, 1);
			ImGui::SameLine();
			ImGui::RadioButton("GPU", &useCPU, 0);

			ImGui::SliderFloat("Speed", &dt_slider_value, 0.0f, 1.0f);
			ImGui::SliderFloat("Inner Temp", &Tin, 0.0f, 100.0f);
			ImGui::SliderFloat("Outer Temp", &Tout, 0.0f, 100.0f);
			ImGui::SliderFloat("Transfer Coeff.", &a, 0.01f, 1.0f);
			ImGui::SliderFloat("Disk Radius", &R, 0.0f, 0.5f);

			if (ImGui::Button("Reset")) {
				initDisk(squares, W, H, dx, dy, Height, Width, R * R, Tin, Tout, &center_coordinates);
				glfwSetTime(0);
				float dt = 0;
				updateTemp_CPU(squares, squares_out, N, W, H, dx*dx, dy*dy, a, dt * dt_slider_value + FLT_MIN);
			}

			if (ImGui::Button("Resume/Pause")) {
				start = !start;
			}
			ImGui::End();
		}

		// Tell OpenGL which Shader Program we want to use
		shaderProgram.Activate();
		// Assigns a value to the uniform; NOTE: Must always be done after activating the Shader Program
		glUniform1f(uniID, 1.0f);
		// Bind the VAO so OpenGL knows to use it
		VAO1.Bind();
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "view"), 1, GL_FALSE, value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "projection"), 1, GL_FALSE, value_ptr(projection));

		drawSquares(squares, &shaderProgram, N);
		// Largest stable time step
		float dt = dx * dx * dy * dy / (2.0 * a * (dx * dx + dy * dy));

		if (start & useCPU) updateTemp_CPU(squares, squares_out, N, W, H, dx*dx, dy*dy, a, dt * dt_slider_value + FLT_MIN);
		if (start & !useCPU) updateTemp_GPU(squares, squares_out, N, W, H, dx*dx, dy*dy, a, dt * dt_slider_value + FLT_MIN);
		// Swap the back buffer with the front buffer
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
	}

	// Delete all the objects we've created
	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
	shaderProgram.Delete();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	// Delete window before ending the program
	glfwDestroyWindow(window);
	// Terminate GLFW before ending the program
	glfwTerminate();
	return 0;
}