// Encapsulates a render result
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_RENDER_BUFFER_H
#define STILLLEBEN_RENDER_BUFFER_H

#include <memory>

namespace sl
{

class RenderBuffer
{
public:

    std::size_t size();
    void copyToMemory(void* dest);
    void copyToCUDA(void* dest);

private:

};

}

#endif
