#include <mutex>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

Mesh::Mesh(const Properties &props)
    : Shape(props), m_surface_area(-1) {
    /* When set to ``true``, Mitsuba will use per-face instead of per-vertex
       normals when rendering the object, which will give it a faceted
       appearance. Default: ``false`` */
    m_vertex_normals = !props.bool_("face_normals", false);
    m_to_world = props.transform("to_world", Transform4f());
}

Mesh::Mesh(const std::string &name, Struct *vertex_struct, Size vertex_count,
           Struct *face_struct, Size face_count)
    : m_name(name), m_vertex_count(vertex_count),
      m_face_count(face_count), m_vertex_struct(vertex_struct),
      m_face_struct(face_struct), m_surface_area(-1) {
    auto check_field = [](const Struct *s, size_t idx,
                          const std::string &suffix_exp,
                          Struct::EType type_exp) {
        if (idx >= s->field_count())
            Throw("Mesh::Mesh(): Incompatible data structure %s", s->to_string());
        auto field = s->operator[](idx);
        std::string suffix = field.name;
        auto it = suffix.rfind(".");
        if (it != std::string::npos)
            suffix = suffix.substr(it + 1);
        if (suffix != suffix_exp || field.type != type_exp)
            Throw("Mesh::Mesh(): Incompatible data structure %s", s->to_string());
    };

    check_field(vertex_struct, 0, "x",  struct_traits<Float>::value);
    check_field(vertex_struct, 1, "y",  struct_traits<Float>::value);
    check_field(vertex_struct, 2, "z",  struct_traits<Float>::value);

    check_field(face_struct,   0, "i0", struct_traits<Index>::value);
    check_field(face_struct,   1, "i1", struct_traits<Index>::value);
    check_field(face_struct,   2, "i2", struct_traits<Index>::value);

    m_vertex_normals = vertex_struct->has_field("nx") &&
                       vertex_struct->has_field("ny") &&
                       vertex_struct->has_field("nz");

    if (m_vertex_normals) {
        check_field(vertex_struct, 3, "nx", Struct::EFloat16);
        check_field(vertex_struct, 4, "ny", Struct::EFloat16);
        check_field(vertex_struct, 5, "nz", Struct::EFloat16);
        m_normals_offset = vertex_struct->field("nx").offset;
    }

    m_vertex_size = (Size) m_vertex_struct->size();
    m_face_size = (Size) m_face_struct->size();

    m_vertices = VertexHolder(
        (uint8_t *) enoki::alloc((vertex_count + 1) * m_vertex_size));
    m_faces = FaceHolder(
        (uint8_t *) enoki::alloc((face_count + 1) * m_face_size));
}

Mesh::~Mesh() { }

void Mesh::write(Stream *) const {
    NotImplementedError("write");
}

BoundingBox3f Mesh::bbox() const {
    return m_bbox;
}

BoundingBox3f Mesh::bbox(Index index) const {
    Assert(index <= m_face_count);

    auto idx = (const Index *) face(index);
    Assert(idx[0] < m_vertex_count && idx[1] < m_vertex_count &&
           idx[2] < m_vertex_count);

    Point3f v0 = load<Point3f>((Float *) vertex(idx[0]));
    Point3f v1 = load<Point3f>((Float *) vertex(idx[1]));
    Point3f v2 = load<Point3f>((Float *) vertex(idx[2]));

    return BoundingBox3f(
        min(min(v0, v1), v2),
        max(max(v0, v1), v2)
    );
}

void Mesh::recompute_vertex_normals() {
    if (unlikely(!m_vertex_normals)) {
        NotImplementedError("Storing new normals in a Mesh that didn't have"
                            " normals at construction.");
    }
    m_normals_offset = m_vertex_struct->field("nx").offset;
    m_vertex_normals = true;

    std::vector<Normal3f, aligned_allocator<Normal3f>> normals(
            m_vertex_count, zero<Normal3f>());
    size_t invalid_counter = 0;
    Timer timer;

    /* Weighting scheme based on "Computing Vertex Normals from Polygonal Facets"
       by Grit Thuermer and Charles A. Wuethrich, JGT 1998, Vol 3 */
    for (Size i = 0; i < m_face_count; ++i) {
        const Index *idx = (const Index *) face(i);
        Assert(idx[0] < m_vertex_count && idx[1] < m_vertex_count && idx[2] < m_vertex_count);
        Point3f v[3] { load<Point3f>((Float *) vertex(idx[0])),
                       load<Point3f>((Float *) vertex(idx[1])),
                       load<Point3f>((Float *) vertex(idx[2])) };

        Vector3f side_0 = v[1] - v[0], side_1 = v[2] - v[0];
        Normal3f n = cross(side_0, side_1);
        Float length_sqr = squared_norm(n);
        if (likely(length_sqr > 0)) {
            n *= rsqrt<Vector3f::Approx>(length_sqr);

            // Use Enoki to compute the face angles at the same time
            auto side_a = transpose(Array<Packet<float, 3>, 3>{ side_0, v[2] - v[1], v[0] - v[2] });
            auto side_b = transpose(Array<Packet<float, 3>, 3>{ side_1, v[0] - v[1], v[1] - v[2] });
            Vector3f face_angles = unit_angle(normalize(side_a), normalize(side_b));

            for (size_t j = 0; j < 3; ++j)
                normals[idx[j]] += n * face_angles[j];
        }
    }

    for (Size i = 0; i < m_vertex_count; i++) {
        Normal3f n = normals[i];
        Float length = norm(n);
        if (likely(length != 0.f)) {
            n /= length;
        } else {
            n = Normal3f(1, 0, 0); // Choose some bogus value
            invalid_counter++;
        }

        store(vertex(i) + m_normals_offset, Normal3h(n));
    }

    if (invalid_counter == 0)
        Log(EDebug, "\"%s\": computed vertex normals (took %s)", m_name,
            util::time_string(timer.value()));
    else
        Log(EWarn, "\"%s\": computed vertex normals (took %s, %i invalid vertices!)",
            m_name, util::time_string(timer.value()), invalid_counter);
}

void Mesh::recompute_bbox() {
    m_bbox.reset();
    for (Size i = 0; i < m_vertex_count; ++i)
        m_bbox.expand(load<Point3f>((Float *) vertex(i)));
}

namespace {
constexpr size_t max_vertices = 10;

size_t sutherland_hodgman(Point3d *input, size_t in_count, Point3d *output,
                          int axis, double split_pos, bool is_minimum) {
    if (in_count < 3)
        return 0;

    Point3d cur        = input[0];
    double sign        = is_minimum ? 1.0 : -1.0;
    double distance    = sign * (cur[axis] - split_pos);
    bool  cur_is_inside = (distance >= 0);
    size_t out_count    = 0;

    for (size_t i=0; i<in_count; ++i) {
        size_t next_idx = i+1;
        if (next_idx == in_count)
            next_idx = 0;

        Point3d next = input[next_idx];
        distance = sign * (next[axis] - split_pos);
        bool next_is_inside = (distance >= 0);

        if (cur_is_inside && next_is_inside) {
            /* Both this and the next vertex are inside, add to the list */
            Assert(out_count + 1 < max_vertices);
            output[out_count++] = next;
        } else if (cur_is_inside && !next_is_inside) {
            /* Going outside -- add the intersection */
            double t = (split_pos - cur[axis]) / (next[axis] - cur[axis]);
            Assert(out_count + 1 < max_vertices);
            Point3d p = cur + (next - cur) * t;
            p[axis] = split_pos; // Avoid roundoff errors
            output[out_count++] = p;
        } else if (!cur_is_inside && next_is_inside) {
            /* Coming back inside -- add the intersection + next vertex */
            double t = (split_pos - cur[axis]) / (next[axis] - cur[axis]);
            Assert(out_count + 2 < max_vertices);
            Point3d p = cur + (next - cur) * t;
            p[axis] = split_pos; // Avoid roundoff errors
            output[out_count++] = p;
            output[out_count++] = next;
        } else {
            /* Entirely outside - do not add anything */
        }
        cur = next;
        cur_is_inside = next_is_inside;
    }
    return out_count;
}
}  // end namespace

BoundingBox3f Mesh::bbox(Index index, const BoundingBox3f &clip) const {
    /* Reserve room for some additional vertices */
    Point3d vertices1[max_vertices], vertices2[max_vertices];
    size_t nVertices = 3;

    Assert(index <= m_face_count);

    auto idx = (const Index *) face(index);
    Assert(idx[0] < m_vertex_count);
    Assert(idx[1] < m_vertex_count);
    Assert(idx[2] < m_vertex_count);

    Point3f v0 = load<Point3f>((Float *) vertex(idx[0]));
    Point3f v1 = load<Point3f>((Float *) vertex(idx[1]));
    Point3f v2 = load<Point3f>((Float *) vertex(idx[2]));

    /* The kd-tree code will frequently call this function with
       almost-collapsed bounding boxes. It's extremely important not to
       introduce errors in such cases, otherwise the resulting tree will
       incorrectly remove triangles from the associated nodes. Hence, do
       the following computation in double precision! */

    vertices1[0] = Point3d(v0);
    vertices1[1] = Point3d(v1);
    vertices1[2] = Point3d(v2);

    for (int axis=0; axis<3; ++axis) {
        nVertices = sutherland_hodgman(vertices1, nVertices, vertices2, axis,
                                      (double) clip.min[axis], true);
        nVertices = sutherland_hodgman(vertices2, nVertices, vertices1, axis,
                                      (double) clip.max[axis], false);
    }

    BoundingBox3f result;
    for (size_t i = 0; i < nVertices; ++i) {
        /* Convert back into floats (with conservative custom rounding modes) */
        Array<double, 3, false, RoundingMode::Up>   p1Up  (vertices1[i]);
        Array<double, 3, false, RoundingMode::Down> p1Down(vertices1[i]);
        result.min = min(result.min, Point3f(p1Down));
        result.max = max(result.max, Point3f(p1Up));
    }
    result.clip(clip);

    return result;
}

template <typename PositionSample, typename Point2>
void Mesh::sample_position_impl(PositionSample &p_rec, const Point2 &sample_,
                                const mask_t<value_t<Point2>> &active) const {
    using Index   = like_t<typename Point2::Value, Shape::Index>;
    using Point3  = typename PositionSample::Point3;
    using Normal3 = typename PositionSample::Normal3;
    using Vector3 = vector3_t<Point3>;

    ensure_table_ready();

    Point2 sample(sample_);
    Index index;
    std::tie(index, sample.y()) = m_area_distribution.sample_reuse(sample.y());

    auto face_ptr = (const typename Shape::Index *) faces();
    index *= m_face_size;
    Index i0 = gather<Index, 1>(face_ptr  , index, active),
          i1 = gather<Index, 1>(face_ptr+1, index, active),
          i2 = gather<Index, 1>(face_ptr+2, index, active);
    Point3 p0  = vertex_position(i0, active),
           p1  = vertex_position(i1, active),
           p2  = vertex_position(i2, active);

    Vector3 side_a = p1 - p0,
            side_b = p2 - p0;

    Point2 bary = warp::square_to_uniform_triangle(sample);
    masked(p_rec.p, active) =  p0 + (side_a * bary.x()) + (side_b * bary.y());

    if (has_vertex_normals()) {
        const Normal3 n0 = vertex_normal(i0, active),
                      n1 = vertex_normal(i1, active),
                      n2 = vertex_normal(i2, active);
        masked(p_rec.n, active) = normalize(n0 * (1.0f - bary.x() - bary.y())
                                          + n1 * bary.x() + n2 * bary.y());
    }
    // TODO: set UV from the UV of the vertices, if available
    masked(p_rec.uv, active) = 0.5f;

    masked(p_rec.pdf, active) = m_inv_surface_area;
    masked(p_rec.measure, active) = EArea;
}
void Mesh::sample_position(PositionSample3f &p_rec,
                           const Point2f &sample) const {
    return sample_position_impl(p_rec, sample, true);
}
void Mesh::sample_position(PositionSample3fP &p_rec,
                           const Point2fP &sample,
                           const mask_t<FloatP> &active) const {
    return sample_position_impl(p_rec, sample, active);
}

void Mesh::prepare_sampling_table() {
    if (m_face_count == 0)
        Throw("Cannot prepare the sampling table of an empty mesh: %s",
              this->to_string());

    std::lock_guard<tbb::spin_mutex> lock(m_mutex);
    if (m_surface_area < 0) {
        // Generate a PDF for sampling wrt. the area of each face.
        m_area_distribution.reserve(m_face_count);
        for (size_t i = 0; i < m_face_count; i++) {
            Float area = face_area(i);
            Assert(area >= 0);
            m_area_distribution.append(area);
        }
        m_surface_area = m_area_distribution.normalize();
        m_inv_surface_area = rcp(m_surface_area);
    }
}


template <typename SurfaceInteraction, typename Mask>
ENOKI_INLINE auto Mesh::normal_derivative_impl(const SurfaceInteraction &si,
                                               bool shading_frame,
                                               const Mask &active) const {
    using Value   = typename SurfaceInteraction::Value;
    using Index   = typename SurfaceInteraction::Index;
    using Point3  = typename SurfaceInteraction::Point3;
    using Vector3 = typename SurfaceInteraction::Vector3;
    using Normal3 = typename SurfaceInteraction::Normal3;

    if (!shading_frame)
        return std::make_pair(zero<Vector3>(), zero<Vector3>());

    auto face_ptr = faces();
    auto prim_index = si.prim_index * m_face_size;

    // TODO: check this, stride and / or indices are probably incorrect.
    Index i0 = gather<Index>(face_ptr,   prim_index, active) * m_vertex_size,
          i1 = gather<Index>(face_ptr+1, prim_index, active) * m_vertex_size,
          i2 = gather<Index>(face_ptr+2, prim_index, active) * m_vertex_size;

    auto vertex_ptr = vertices(),
         normal_ptr = vertex_ptr + sizeof(Vector3f);

    /// These gathers should turn into loads
    Point3 p0 = gather<Point3>(vertex_ptr, i0, active),
           p1 = gather<Point3>(vertex_ptr, i1, active),
           p2 = gather<Point3>(vertex_ptr, i2, active);

    Normal3 n0 = gather<Normal3>(normal_ptr, i0, active),
            n1 = gather<Normal3>(normal_ptr, i1, active),
            n2 = gather<Normal3>(normal_ptr, i2, active);

    Vector3 rel = si.p - p0,
            du  = p1   - p0,
            dv  = p2   - p0;

    /* Solve a least squares problem to determine
       the UV coordinates within the current triangle */
    Value b1  = dot(du, rel), b2 = dot(dv, rel),
          a11 = dot(du, du), a12 = dot(du, dv),
          a22 = dot(dv, dv),
          inv_det = rcp(a11 * a22 - a12 * a12);

    Value u = ( a22 * b1 - a12 * b2) * inv_det,
          v = (-a12 * b1 + a11 * b2) * inv_det,
          w = 1.f - u - v;

    /* Now compute the derivative of "normalize(u*n1 + v*n2 + (1-u-v)*n0)"
       with respect to [u, v] in the local triangle parameterization.

       Since d/du [f(u)/|f(u)|] = [d/du f(u)]/|f(u)|
         - f(u)/|f(u)|^3 <f(u), d/du f(u)>, this results in
    */

    Normal3 N(u * n1 + v * n2 + w * n0);
    Value il = rsqrt<true>(squared_norm(N));
    N *= il;

    Vector3 dndu = (n1 - n0) * il;
    Vector3 dndv = (n2 - n0) * il;

    dndu -= N * dot(N, dndu);
    dndv -= N * dot(N, dndv);

    return std::make_pair(dndu, dndv);
}
std::pair<Vector3f, Vector3f>
Mesh::normal_derivative(const SurfaceInteraction3f &si, bool shading_frame) const {
    return normal_derivative_impl<SurfaceInteraction3f>(si, shading_frame, true);
}

std::pair<Vector3fP, Vector3fP>
Mesh::normal_derivative(const SurfaceInteraction3fP &si, bool shading_frame,
                        const mask_t<FloatP> &active) const {
    return normal_derivative_impl<SurfaceInteraction3fP>(si, shading_frame, active);
}

std::string Mesh::to_string() const {
    std::ostringstream oss;
    oss << class_()->name() << "[" << std::endl
        << "  name = \"" << m_name << "\"," << std::endl
        << "  bbox = " << string::indent(m_bbox) << "," << std::endl
        << "  vertex_struct = " << string::indent(m_vertex_struct) << "," << std::endl
        << "  vertex_count = " << m_vertex_count << "," << std::endl
        << "  vertices = [" << util::mem_string(m_vertex_size * m_vertex_count) << " of vertex data]," << std::endl
        << "  vertex_normals = " << m_vertex_normals << "," << std::endl
        << "  face_struct = " << string::indent(m_face_struct) << "," << std::endl
        << "  face_count = " << m_face_count << "," << std::endl
        << "  faces = [" << util::mem_string(m_face_size * m_face_count) << " of face data]" << std::endl
        << "]";
    return oss.str();
}

template MTS_EXPORT_RENDER auto Mesh::intersect_face(Index, const Ray3f &) const;
template MTS_EXPORT_RENDER auto Mesh::intersect_face(Index, const Ray3fP &) const;

template MTS_EXPORT_RENDER auto Mesh::vertex_normal(const Index &, const bool &) const;
template MTS_EXPORT_RENDER auto Mesh::vertex_normal(const UInt32P  &, const mask_t<UInt32P> &) const;

MTS_IMPLEMENT_CLASS(Mesh, Shape)
NAMESPACE_END(mitsuba)
