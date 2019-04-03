// Mesh simplification
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_SIMPLIFY_MESH_H
#define STILLLEBEN_SIMPLIFY_MESH_H

#include <Magnum/GL/Mesh.h>

#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector3.h>

#include <algorithm>
#include <cstring>
#include <fstream>

//
// This code is largely 1:1 from
// https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
// (licensed under MIT)
//
// I ported to Magnum maths and adapted to slightly more modern C++, although
// the algorithm is still quite low-level and does a lot of index fiddling.
//

namespace sl
{
namespace mesh_tools
{

template<class Vertex>
class QuadricEdgeSimplification
{
public:
    QuadricEdgeSimplification(
        std::vector<Magnum::UnsignedInt>& indices,
        std::vector<Vertex>& vertices
    )
     : m_indices{indices}
     , m_vertices{vertices}
     , m_triangleDeleted(static_cast<std::size_t>(indices.size()/3), false)
     , m_normals(static_cast<std::size_t>(indices.size()/3))
     , m_Q(vertices.size(), Magnum::Matrix4{Magnum::Math::ZeroInit})
     , m_error(indices.size(), 0.0)
     , m_border(vertices.size(), false)
     , m_tstart(vertices.size())
     , m_tcount(vertices.size())
     , m_dirty(indices.size()/3, false)
    {}

    void simplify(std::size_t targetTriangles, double aggressiveness=5)
    {
        using namespace Magnum;

        const UnsignedInt numTriangles = m_indices.size()/3;

        std::size_t deletedTriangles = 0;

        std::vector<std::size_t> deleted0, deleted1;

        checkMesh();

        for(UnsignedInt iter = 0; iter < 2; ++iter)
        {
            printf("iter triangles=%lu, vertices=%lu\n", numTriangles - deletedTriangles, m_vertices.size());
            if(numTriangles - deletedTriangles <= targetTriangles)
                break;

            // Update mesh once in a while
//             if(iter % 5 == 0)
                updateMesh(iter == 0);

            std::fill(m_dirty.begin(), m_dirty.end(), false);

            // triangles with edge errors *below* this threshold will be
            // removed
            const float threshold = 1e-9 * std::pow<float>(
                iter + 3, aggressiveness
            );

            // remove vertices & mark deleted triangles
            for(std::size_t i = 0; i < numTriangles; ++i)
            {
                if(m_triangleDeleted[i] || m_dirty[i])
                    continue;

                for(std::size_t j = 0; j < 3; ++j)
                {
                    if(m_error[i*3+j] >= threshold)
                        continue;

                    auto idx0 = i*3 + j;
                    auto idx1 = i*3 + ((j+1) % 3);
                    auto v0 = m_indices[idx0];
                    auto v1 = m_indices[idx1];

                    // Border check
                    if(m_border[v0] != m_border[v1])
                        continue;

                    // Compute vertex to collapse to
                    Vector3 p;
                    calculateError(v0, v1, p);

                    deleted0.resize(m_tcount[v0]);
                    deleted1.resize(m_tcount[v1]);

                    // Do not remove edge if that would flip the triangle
                    if(flipped(p, idx0, idx1, v0, v1, deleted0))
                        continue;
                    if(flipped(p, idx1, idx0, v1, v0, deleted1))
                        continue;

                    // not flipped, so remove the edge.
                    m_vertices[v0] = p;
                    m_Q[v0] += m_Q[v1];

                    auto tstart = m_ref_triangle.size();

                    updateTriangles(idx0, v0, deleted0, deletedTriangles);
                    updateTriangles(idx0, v1, deleted1, deletedTriangles);

                    auto tcount = m_ref_triangle.size() - tstart;

                    if(tcount <= m_tcount[v0])
                    {
                        // save RAM
                        if(tcount)
                        {
                            std::memcpy(
                                &m_ref_triangle[m_tstart[v0]],
                                &m_ref_triangle[tstart],
                                tcount * sizeof(std::size_t)
                            );
                            std::memcpy(
                                &m_ref_vertex[m_tstart[v0]],
                                &m_ref_vertex[tstart],
                                tcount * sizeof(std::size_t)
                            );
                        }
                    }
                    else
                    {
                        // append
                        m_tstart[v0] = tstart;
                    }

                    m_tcount[v0] = tcount;
                    break;
                }

                if(numTriangles - deletedTriangles <= targetTriangles)
                    break;
            }
        }

        compactMesh();
        checkMesh();
    }

    void writeOBJ(const std::string& filename)
    {
        std::ofstream out(filename);
        if(!out)
            throw std::runtime_error("Can't write to output file");

        out.imbue(std::locale::classic());

        for(auto& v : m_vertices)
            out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";

        for(std::size_t i = 0; i < m_indices.size()/3; ++i)
        {
            out << "f";
            for(std::size_t j = 0; j < 3; ++j)
            {
                out << " " << m_indices[i*3+j]+1;
            }
            out << "\n";
        }
    }
private:
    void checkMesh()
    {
        for(std::size_t v : m_indices)
        {
            if(v >= m_vertices.size())
            {
                throw std::logic_error("Mesh indices invalid");
            }
        }

        for(auto& v : m_vertices)
        {
            if(!std::isfinite(v.x()) || !std::isfinite(v.y()) || !std::isfinite(v.z()))
            {
                throw std::runtime_error("Got non-finite vertex");
            }
        }
    }

    void compactTriangles()
    {
        std::size_t n = m_indices.size()/3;
        std::size_t dst = 0;
        for(std::size_t i = 0; i < n; ++i)
        {
            if(!m_triangleDeleted[i])
            {
                for(std::size_t j = 0; j < 3; ++j)
                {
                    m_indices[3*dst + j] = m_indices[3*i + j];
                    m_error[3*dst + j] = m_error[3*i + j];
                }
                m_normals[dst] = m_normals[i];

                dst++;
            }
        }

        m_indices.resize(3*dst);
        m_error.resize(3*dst);
        m_normals.resize(dst);
        m_triangleDeleted.resize(dst);
        std::fill(m_triangleDeleted.begin(), m_triangleDeleted.end(), false);
    }

    static Magnum::Matrix4 planeMatrix(const Magnum::Vector3& normal, float d)
    {
        using namespace Magnum;

        Vector4 v{normal, d};
        Matrix4 ret{Magnum::Math::NoInit};

        for(std::size_t i = 0; i < 4; ++i)
        {
            for(std::size_t j = 0; j < 4; ++j)
            {
                ret[i][j] = v[i] * v[j];
            }
        }
        return ret;
    }

    float vertexError(const Magnum::Matrix4& Q, const Magnum::Vector3& p) const
    {
        using namespace Magnum;

        auto ph = Vector4{p, 1.0};
        return (
            Math::RectangularMatrix<4,1, float>::fromVector(ph)
            * Math::RectangularMatrix<4,4, float>{Q}
            * Math::RectangularMatrix<1,4, float>::fromVector(ph)
        )[0][0];
    }

    float calculateError(std::size_t idx1, std::size_t idx2, Magnum::Vector3& result) const
    {
        using namespace Magnum;

        // Compute interpolated vertex
        // We are looking for the solution to Qhat * p = [0 0 0 1]
        // where Qhat is Q with Q[3,:] = [0 0 0 1].

        Matrix4 Q = m_Q[idx1] + m_Q[idx2];
        bool border = m_border[idx1] && m_border[idx2];

        Matrix3 Qd = Q.rotationScaling(); // upper left 3x3 submatrix

        float det = Qd.determinant();

        if(std::abs(det) > 1e-7 && !border)
        {
            // Qd is invertible
            result = Qd.inverted() * (-Q[3].xyz());
            if(!std::isfinite(result.x()) || !std::isfinite(result.y()) || !std::isfinite(result.z()))
            {
                Corrade::Utility::Error{} << "Obtained non-finite number from optimization";
                Corrade::Utility::Error{} << "Qd";
                Corrade::Utility::Error{} << Qd;
                Corrade::Utility::Error{} << "result:" << result;
                throw std::runtime_error("Quadric optimization failed");
            }
            return vertexError(Q, result);
        }
        else
        {
            // Qd is not invertible -> try to find best result
            const Vector3& p1 = m_vertices[idx1];
            const Vector3& p2 = m_vertices[idx2];
            auto p3 = (p1 + p2) / 2.0f;

            std::array<std::pair<float, std::reference_wrapper<const Vector3>>, 3> errors{{
                {vertexError(Q, p1), std::cref(p1)},
                {vertexError(Q, p2), std::cref(p2)},
                {vertexError(Q, p3), std::cref(p3)}
            }};
            auto min_pair = std::min_element(errors.begin(), errors.end(),
                [](auto a, auto b){
                    return a.first < b.first;
                }
            );

            result = min_pair->second;
            return min_pair->first;
        }
    }

    void initQuadrics()
    {
        using namespace Magnum;

        std::size_t n = m_indices.size()/3;

        for(std::size_t i = 0; i < m_vertices.size(); ++i)
            m_Q[i] = Matrix4(Math::ZeroInit);

        // Init quadrics
        for(std::size_t i = 0; i < n; ++i)
        {
            Vector3 triangle[3] = {
                m_vertices[m_indices[3*i+0]],
                m_vertices[m_indices[3*i+1]],
                m_vertices[m_indices[3*i+2]]
            };

            Vector3 normal = Magnum::Math::cross(
                triangle[1] - triangle[0],
                triangle[2] - triangle[0]
            ).normalized();

            m_normals[i] = normal;

            for(std::size_t j = 0; j < 3; ++j)
            {
                m_Q[m_indices[3*i+j]] += planeMatrix(
                    normal,
                    -Magnum::Math::dot(normal, triangle[0])
                );
            }
        }

        // Calculate edge error
        Magnum::Vector3 p;
        for(std::size_t i = 0; i < n; ++i)
        {
            for(std::size_t j = 0; j < 3; ++j)
            {
                m_error[3*i+j] = calculateError(
                    m_indices[3*i+j], m_indices[3*i+((j+1)%3)], p
                );
            }
        }
    }

    void updateMesh(bool init = false)
    {
        printf("updateMesh(%d)\n", init);

        if(!init)
            compactTriangles();

//         if(init)
            initQuadrics();

        std::fill(m_tstart.begin(), m_tstart.end(), 0);
        std::fill(m_tcount.begin(), m_tcount.end(), 0);

        for(std::size_t i = 0; i < m_indices.size(); ++i)
            m_tcount[m_indices[i]]++;

        std::size_t tstart = 0;
        for(std::size_t i = 0; i < m_vertices.size(); ++i)
        {
            m_tstart[i] = tstart;
            tstart += m_tcount[i];
            m_tcount[i] = 0; // why?
        }

        m_ref_triangle.resize(m_indices.size());
        m_ref_vertex.resize(m_indices.size());
        for(std::size_t i = 0; i < m_indices.size(); ++i)
        {
            auto v = m_indices[i];
            auto refID = m_tstart[v] + m_tcount[v];
            m_tcount[v]++;

            m_ref_triangle[refID] = i / 3;
            m_ref_vertex[refID] = i % 3;
        }

        // Identify boundary
        if(init)
        {
            m_border.clear();
            m_border.resize(m_vertices.size(), false);

            std::vector<std::size_t> vcount, vids;

            for(std::size_t i = 0; i < m_vertices.size(); ++i)
            {
                std::size_t tstart = m_tstart[i];
                std::size_t tcount = m_tcount[i];

                vcount.clear();
                vids.clear();

                for(std::size_t j = 0; j < tcount; ++j)
                {
                    auto triangle = m_ref_triangle[tstart+j];
                    for(std::size_t k = 0; k < 3; ++k)
                    {
                        auto v = m_indices[triangle*3 + k];

                        auto it = std::find(vids.begin(), vids.end(), v);

                        if(it == vids.end())
                        {
                            vcount.push_back(1);
                            vids.push_back(v);
                        }
                        else
                            vcount[it - vids.begin()]++;
                    }
                }

                for(std::size_t j = 0; j < vids.size(); ++j)
                {
                    if(vcount[j] == 1)
                        m_border[vids[j]] = true;
                }
            }
        }
    }

    // Check if a triangle flips when this edge is removed
    bool flipped(const Magnum::Vector3& p, std::size_t idx0, std::size_t idx1,
        std::size_t v0, std::size_t v1, std::vector<std::size_t>& deleted)
    {
        auto tstart = m_tstart[v0];
        auto tcount = m_tcount[v0];

        for(std::size_t i = 0; i < tcount; ++i)
        {
            auto triangle = m_ref_triangle[tstart + i];

            if(m_triangleDeleted[triangle])
                continue;

            auto s = m_ref_vertex[tstart + i];
            auto id1 = m_indices[triangle*3 + ((s+1) % 3)];
            auto id2 = m_indices[triangle*3 + ((s+2) % 3)];

            if(id1 == v1 || id2 == v1)
            {
                deleted[i] = true;
                continue;
            }

            auto d1 = (m_vertices[id1] - p).normalized();
            auto d2 = (m_vertices[id2] - p).normalized();

            if(std::abs(Magnum::Math::dot(d1, d2)) > 0.999f)
                return true;

            auto n = Magnum::Math::cross(d1, d2).normalized();

            deleted[i] = false;

            if(Magnum::Math::dot(n, m_normals[triangle]) < 0.2)
                return true;
        }

        return false;
    }

    void updateTriangles(std::size_t idx0, std::size_t v,
        const std::vector<std::size_t>& deleted, std::size_t& deletedTriangles)
    {
        Magnum::Vector3 p;

        const auto tstart = m_tstart[v];
        const auto tcount = m_tcount[v];

        for(std::size_t i = 0; i < tcount; ++i)
        {
            auto triangle = m_ref_triangle[tstart + i];

            if(m_triangleDeleted[triangle])
                continue;

            if(deleted[i])
            {
                m_triangleDeleted[triangle] = true;
                deletedTriangles++;
                continue;
            }

            m_indices[triangle*3 + m_ref_vertex[tstart + i]] = v;
            m_dirty[triangle] = true;

            auto v0 = m_indices[triangle*3 + 0];
            auto v1 = m_indices[triangle*3 + 1];
            auto v2 = m_indices[triangle*3 + 2];

            m_error[triangle*3 + 0] = calculateError(v0, v1, p);
            m_error[triangle*3 + 1] = calculateError(v1, v2, p);
            m_error[triangle*3 + 2] = calculateError(v2, v0, p);

            m_ref_triangle.push_back(triangle);
            m_ref_vertex.push_back(m_ref_vertex[tstart + i]);
        }
    }

    void compactMesh()
    {
        // Abuse the tcount array to indicate whether a vertex is in use
        std::fill(m_tcount.begin(), m_tcount.end(), 0);

        // Compact triangles (m_indices)
        {
            const std::size_t numTriangles = m_indices.size() / 3;
            std::size_t triangleIdx = 0;

            for(std::size_t i = 0; i < numTriangles; ++i)
            {
                if(m_triangleDeleted[i])
                    continue;

                for(std::size_t j = 0; j < 3; ++j)
                {
                    m_indices[triangleIdx*3 + j] = m_indices[i*3 + j];
                    m_tcount[m_indices[i*3+j]] = 1;
                }
                triangleIdx++;
            }
            m_indices.resize(triangleIdx*3);
        }

        // Compact vertices (m_vertices)
        {
            const std::size_t numVertices = m_vertices.size();
            std::size_t vertexIdx = 0;

            // We abuse the m_tstart array here to record the new vertex ID
            for(std::size_t i = 0; i < numVertices; ++i)
            {
                if(!m_tcount[i])
                    continue;

                m_tstart[i] = vertexIdx;
                m_vertices[vertexIdx] = m_vertices[i];

                vertexIdx++;
            }
            m_vertices.resize(vertexIdx);

            // Repair indices
            const std::size_t numTriangles = m_indices.size() / 3;
            for(std::size_t i = 0; i < numTriangles; ++i)
            {
                for(std::size_t j = 0; j < 3; ++j)
                {
                    m_indices[i*3 + j] = m_tstart[m_indices[i*3 + j]];
                }
            }
        }
    }

    std::vector<Magnum::UnsignedInt>& m_indices;
    std::vector<Vertex>& m_vertices;

    std::vector<bool> m_triangleDeleted;
    std::vector<Magnum::Vector3> m_normals;
    std::vector<Magnum::Matrix4> m_Q;

    //! Error per triangle vertex
    std::vector<float> m_error;

    //! Is a vertex a border vertex?
    std::vector<bool> m_border;

    //! At which reference ID does this vertex start?
    std::vector<std::size_t> m_tstart;

    //! how many triangles is this vertex part of?
    std::vector<std::size_t> m_tcount;

    // Reference ID -> triangle
    std::vector<std::size_t> m_ref_triangle;

    // Reference ID -> vertex in triangle (0-2)
    std::vector<std::size_t> m_ref_vertex;

    std::vector<bool> m_dirty;
};

}
}

#endif
