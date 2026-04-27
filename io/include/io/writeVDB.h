#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

class VDBWriter {
    struct Impl;
    const std::unique_ptr<Impl> impl;

    template <class T, size_t N>
    struct AddGridImpl {
        VDBWriter*  that;
        std::string name;
        const void* base;
        uint32_t    sizex, sizey, sizez;
        int32_t     minx, miny, minz;
        uint32_t    pitchx, pitchy, pitchz;

        void operator()() const;
    };

  public:
    VDBWriter();
    ~VDBWriter();

    VDBWriter(const VDBWriter&)            = delete;
    VDBWriter& operator=(const VDBWriter&) = delete;
    VDBWriter(VDBWriter&&)                 = delete;
    VDBWriter& operator=(VDBWriter&&)      = delete;

    template <class T, size_t N, bool normalizedCoords = true>
    void addGrid(const std::string& name, const void* base, uint32_t sizex, uint32_t sizey, uint32_t sizez,
                 uint32_t pitchx = 0, uint32_t pitchy = 0, uint32_t pitchz = 0) {
        if (pitchx == 0) pitchx = sizeof(T) * N;
        if (pitchy == 0) pitchy = pitchx * sizex;
        if (pitchz == 0) pitchz = pitchy * sizey;
        int32_t minx = normalizedCoords ? -(int32_t)sizex / 2 : 0;
        int32_t miny = normalizedCoords ? -(int32_t)sizey / 2 : 0;
        int32_t minz = normalizedCoords ? -(int32_t)sizez / 2 : 0;
        AddGridImpl<T, N>{this, name, base, sizex, sizey, sizez, minx, miny, minz, pitchx, pitchy, pitchz}();
    }

    void write(const std::string& path);
};

#ifdef WRITEVDB_IMPLEMENTATION
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

struct VDBWriter::Impl {
    openvdb::GridPtrVec grids;
};

VDBWriter::VDBWriter() : impl(std::make_unique<Impl>()) {}

VDBWriter::~VDBWriter() = default;

void VDBWriter::write(const std::string& path) { openvdb::io::File(path).write(impl->grids); }

namespace {

template <class T, size_t N>
struct vdbtraits {};

template <>
struct vdbtraits<float, 1> {
    using type = openvdb::FloatGrid;
};

template <>
struct vdbtraits<float, 3> {
    using type = openvdb::Vec3fGrid;
};

// template <>
// struct vdbtraits<double, 1> {
//     using type = openvdb::Vec3DTree;
// };

// template <>
// struct vdbtraits<double, 3> {
//     using type = openvdb::Vec3DGrid;
// };

template <class VecT, class T, size_t... Is>
VecT help_make_vec(const T* ptr, std::index_sequence<Is...>) {
    return VecT(ptr[Is]...);
}

} // namespace

template <class T, size_t N>
void VDBWriter::AddGridImpl<T, N>::operator()() const {
    using GridT = typename vdbtraits<T, N>::type;

    openvdb::tools::Dense<typename GridT::ValueType> dens(openvdb::Coord(sizex, sizey, sizez),
                                                          openvdb::Coord(minx, miny, minz));
    for (uint32_t z = 0; z < sizez; z++) {
        for (uint32_t y = 0; y < sizey; y++) {
            for (uint32_t x = 0; x < sizex; x++) {
                auto ptr = reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + pitchx * x + pitchy * y
                                                      + pitchz * z);
                dens.setValue(x, y, z, help_make_vec<typename GridT::ValueType>(ptr, std::make_index_sequence<N>{}));
            }
        }
    }

    auto                      grid = GridT::create();
    typename GridT::ValueType tolerance{0};
    openvdb::tools::copyFromDense(dens, grid->tree(), tolerance);

    openvdb::MetaMap& meta = *grid;
    meta.insertMeta(openvdb::Name("name"), openvdb::TypedMetadata<std::string>(name));
    that->impl->grids.push_back(grid);
}

// template struct VDBWriter::AddGridImpl<double, 1>;
// template struct VDBWriter::AddGridImpl<double, 3>;
template struct VDBWriter::AddGridImpl<float, 1>;
template struct VDBWriter::AddGridImpl<float, 3>;
#endif
