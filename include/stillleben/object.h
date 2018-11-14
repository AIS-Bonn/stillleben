// Scene object
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_OBJECT_H
#define STILLLEBEN_OBJECT_H

#include <stillleben/math.h>

#include <memory>

namespace sl
{

class Mesh;

class Object
{
public:
    class Private;

    Object();
    ~Object();

    static std::shared_ptr<Object> instantiate(const std::shared_ptr<Mesh>& mesh);

    void setPose(const PoseMatrix& pose);
    PoseMatrix pose() const;

private:
    std::unique_ptr<Private> m_d;
};

}

#endif
