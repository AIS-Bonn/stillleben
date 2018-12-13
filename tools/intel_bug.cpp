
#include <GL/glew.h>

#define GLEW_STATIC

#include <GLFW/glfw3.h>

#include <cstdio>
#include <vector>
#include <string>
#include <map>

const char* VERTEX_SHADER = R"EOS(
#version 450
layout (location = 0) in vec2 position;

void main() {
  gl_Position.xywz = vec4(position, 1.0, 0.0);
}
)EOS";

const char* FRAGMENT_SHADER_1 = R"EOS(
#version 450
layout (location = 0) out uint my_output;
void main() {
  my_output = 5u;
}
)EOS";

const char* FRAGMENT_SHADER_2 = R"EOS(
#version 450
uniform usampler2DMS sampler;
out uint my_output;
void main() {
  my_output = texelFetch(sampler, ivec2(0,0), 0).r;
}
)EOS";

GLuint compileShader(const std::string& shader, GLenum type, const char* source)
{
    GLuint shader_obj = glCreateShader(type);
    glShaderSource(shader_obj, 1, &source, NULL);
    glCompileShader(shader_obj);

    GLint success = 0;
    glGetShaderiv(shader_obj, GL_COMPILE_STATUS, &success);
    if(success == GL_FALSE)
    {
        fprintf(stderr, "Could not compile %s\n", shader.c_str());

        GLint maxLength = 0;
        glGetShaderiv(shader_obj, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(shader_obj, maxLength, &maxLength, &infoLog[0]);

        fprintf(stderr, "Output:\n%s\n", infoLog.data());

        std::abort();
    }

    return shader_obj;
}

void errorCallback(
    GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
            ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
                type, severity, message );
}

const std::map<std::string, GLenum> KNOWN_FORMATS{
    {"GL_R32F", GL_R32F},
    {"GL_R32UI", GL_R32UI},
    {"GL_R16UI", GL_R16UI},
    {"GL_R8UI", GL_R8UI}
};

int main(int argc, char** argv)
{
    if(argc < 2 || std::string(argv[1]) == "--help")
    {
        fprintf(stderr, "Usage: %s <format>\n", argv[0]);
        fprintf(stderr, "Where format is one of: GL_R32F, GL_R32UI, GL_R8UI, GL_R16UI\n");
        return 1;
    }

    auto it = KNOWN_FORMATS.find(argv[1]);
    if(it == KNOWN_FORMATS.end())
    {
        fprintf(stderr, "Unknown format '%s'\n", argv[1]);
        return 1;
    }

    printf("Testing format %s\n", argv[1]);
    GLenum format = it->second;

    if(!glfwInit())
    {
        fprintf(stderr, "glfwInit failed\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    GLFWwindow* window = glfwCreateWindow(640, 480, "My Title", NULL, NULL);
    if(!window)
    {
        fprintf(stderr, "Could not create GLFW window\n");
        return 1;
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        printf("Failed to initialize GLEW.\n");
        glfwTerminate();
        return 1;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(errorCallback, 0);

    // Compile shaders
    GLuint vertex_shader;
    GLuint fragment_shader_1;
    GLuint fragment_shader_2;
    GLuint program1;
    GLuint program2;
    {
        vertex_shader = compileShader("vertex shader", GL_VERTEX_SHADER, VERTEX_SHADER);
        fragment_shader_1 = compileShader("fragment shader 1", GL_FRAGMENT_SHADER, FRAGMENT_SHADER_1);
        fragment_shader_2 = compileShader("fragment shader 2", GL_FRAGMENT_SHADER, FRAGMENT_SHADER_2);

        program1 = glCreateProgram();
        glAttachShader(program1, vertex_shader);
        glAttachShader(program1, fragment_shader_1);
        glLinkProgram(program1);

        program2 = glCreateProgram();
        glAttachShader(program2, vertex_shader);
        glAttachShader(program2, fragment_shader_2);
        glLinkProgram(program2);
    }

    const float quadPoints[]{
        1.0f, -1.0f,
        1.0f, 1.0f,
        -1.0f, -1.0f,
        -1.0f, 1.0f
    };

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 8*sizeof(float), quadPoints, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // First render: fill multisample texture with value "5"
    GLuint texture;
    {
        GLuint fbo;
        glCreateFramebuffers(1, &fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // Create multisample texture
        glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, 1, &texture);
        glTextureStorage2DMultisample(texture, 4, format, 640, 480, GL_FALSE);

        glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, texture, 0);

        const GLenum buffers[]{GL_COLOR_ATTACHMENT0};
        glNamedFramebufferDrawBuffers(fbo, 1, buffers);

        GLenum status = glCheckNamedFramebufferStatus(fbo, GL_DRAW_FRAMEBUFFER);
        if(status != GL_FRAMEBUFFER_COMPLETE)
        {
            fprintf(stderr, "Invalid framebuffer status\n");
            return 1;
        }

        glUseProgram(program1);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    // Second render: read texel from multisample texture
    GLuint outputTexture;
    {
        GLuint fbo;
        glCreateFramebuffers(1, &fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // Create output texture
        glCreateTextures(GL_TEXTURE_RECTANGLE, 1, &outputTexture);
        glTextureStorage2D(outputTexture, 1, GL_R8UI, 640, 480);
        glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, outputTexture, 0);

        const GLenum buffers[]{GL_COLOR_ATTACHMENT0};
        glNamedFramebufferDrawBuffers(fbo, 1, buffers);

        // bind input texture
        glBindTextureUnit(0, texture);

        glUseProgram(program2);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    // Read out outputTexture
    {
        std::vector<uint8_t> data(640*480);

        glGetTextureImage(outputTexture, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, data.size(), data.data());

        printf("First pixels (these should have value 5):\n");
        for(std::size_t i = 0; i < 100; ++i)
        {
            printf("%02X ", data[i]);
        }
        printf("\n");

        bool correct = true;
        for(auto val : data)
        {
            if(val != 5)
            {
                correct = false;
                break;
            }
        }

        if(correct)
            printf("PASS\n");
        else
            printf("FAIL\n");
    }

    glfwTerminate();
    return 0;
}
