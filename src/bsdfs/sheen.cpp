#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Sheen final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, MicrofacetDistribution)

    static constexpr auto Pi    = math::Pi<scalar_t<Float>>;
    static constexpr auto InvPi = math::InvPi<scalar_t<Float>>;

    Sheen(const Properties &props) : Base(props) {
        // if (props.has_property("specular_reflectance"))
        //     m_specular_reflectance   = props.texture<Texture>("specular_reflectance", 1.f);

        m_alpha = props.texture<Texture>("alpha");
        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;

        parameters_changed();
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {
    }

    Float sqr(Float a) const {
        return a*a;
    }

    Float L(Float cosT, Float a, Float b, Float c, Float d, Float e) const{
        return a/(1.0f+b*enoki::pow(cosT,c)) + d*cosT + e;    
    }

    Float eval_Lambda(Float cosT, Float alpha) const {
        Float a = enoki::lerp(21.5473f, 25.3245f, sqr(1-alpha));
        Float b = enoki::lerp(3.82987f, 3.32435f, sqr(1-alpha));
        Float c = enoki::lerp(0.19823f, 0.16801f, sqr(1-alpha));
        Float d = enoki::lerp(-1.97760f, -1.27393f, sqr(1-alpha));
        Float e = enoki::lerp(-4.32054f, -4.85967f, sqr(1-alpha));                

        return select(cosT<0.5f, enoki::exp(L(cosT,a,b,c,d,e)), enoki::exp(2.0f*L(0.5f,a,b,c,d,e)-L(1.0f-cosT,a,b,c,d,e)));
    }

    Spectrum eval_F(Float HdotV) const {
        return Spectrum(1.0f);
    }

    Float eval_D(Float HdotN, Float alpha) const {
        return (2.0f + 1.0f/alpha) * enoki::pow(enoki::sin(enoki::acos(HdotN)), 1.0f/alpha) * (0.5f*InvPi);
    }

    Float eval_G(Float NdotV, Float NdotL, Float alpha) const {
        return 1.0f / (1.0f + eval_Lambda(NdotV, alpha) + eval_Lambda(NdotL, alpha));
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        BSDFSample3f bs = zero<BSDFSample3f>();
        Float NdotV = Frame3f::cos_theta(si.wi);

        // Ignore perfectly grazing configurations
        active &= NdotV > 0.f;
        if (unlikely(none_or<false>(active) ||
            !ctx.is_enabled(BSDFFlags::GlossyReflection))) {
            return { bs, 0.f };
        }

        // Perfect specular reflection based on the microfacet normal
        bs.wo = warp::square_to_uniform_hemisphere(sample2);
        bs.pdf = warp::square_to_uniform_hemisphere_pdf(bs.wo);
        bs.eta = 1.f;
        bs.sampled_component = 0;
        bs.sampled_type = +BSDFFlags::GlossyReflection;

        // Calculate the half-direction vector
        Vector3f h = normalize(bs.wo + si.wi);
        Float HdotV = dot(si.wi, h);
        Float HdotN = Frame3f::cos_theta(h);
        Float NdotL = Frame3f::cos_theta(bs.wo);
        Float alpha = m_alpha->eval_1(si, active);
        alpha = max(alpha, 1.0E-8);

        Spectrum value = eval_F(HdotV)*eval_D(HdotN, alpha)*eval_G(NdotV, NdotL, alpha) / (4.0f * bs.pdf * NdotV);
        return { bs, select(active && bs.pdf > 0.f, unpolarized<Spectrum>(value), 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        const Float NdotV = Frame3f::cos_theta(si.wi);
        const Float NdotL = Frame3f::cos_theta(wo);

        // Ignore direction below the horizon
        active &= NdotV > 0.0f;
        active &= NdotL > 0.0f;

        // Compute the half-vector
        Vector3f h = normalize(si.wi + wo);
        Float HdotN = Frame3f::cos_theta(h);
        Float HdotV = dot(h, si.wi);
        Float alpha = m_alpha->eval_1(si, active);
        alpha = max(alpha, 1.0E-8);

        Spectrum value = eval_F(HdotV)*eval_D(HdotN, alpha)*eval_G(NdotV, NdotL, alpha) / (4.0f * NdotV);
        return select(active, unpolarized<Spectrum>(value), 0.0f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        // Only account for GlossyReflection tag
        if (!ctx.is_enabled(BSDFFlags::GlossyReflection))
            return 0.f;

        const Float NdotV = Frame3f::cos_theta(si.wi);
        const Float NdotL = Frame3f::cos_theta(wo);
        
        // Ignore direction below the horizon
        active &= NdotV > 0.0f;
        active &= NdotL > 0.0f;

        Float pdf = warp::square_to_uniform_hemisphere_pdf(wo);

        return select(active, pdf, 0.0f);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("alpha", m_alpha.get());
        // if (m_specular_reflectance)
        //     callback->put_object("specular_reflectance", m_specular_reflectance.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Sheen[" << std::endl
            << "  alpha = " << string::indent(m_alpha) << "," << std::endl
        // if (m_specular_reflectance)
        //     oss << "  specular_reflectance = "   << string::indent(m_specular_reflectance) << "," << std::endl;
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    // ref<Texture> m_specular_reflectance;
    ref<Texture> m_alpha;
};

MTS_IMPLEMENT_CLASS_VARIANT(Sheen, BSDF)
MTS_EXPORT_PLUGIN(Sheen, "Sheen")
NAMESPACE_END(mitsuba)
