//! HIP/ROCm GPU kernels for Candle, compiled to HSACO code objects.
//!
//! Each kernel family is compiled at build time via `hipcc --genco` and
//! embedded as a binary blob. At runtime, the HIP backend loads these
//! via `hipModuleLoadData` and looks up individual functions by name.

mod hsaco {
    include!(concat!(env!("OUT_DIR"), "/hsaco.rs"));
}

/// Identifies a kernel module for lazy-loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    AttnQ8Kv,
    Binary,
    Cast,
    Conv,
    Fill,
    FlashAttn,
    FlashAttnV2,
    FusedFfnDecode,
    GatedDeltaNet,
    Indexing,
    MmqTurbo,
    Quantized,
    QuantizeKvQ8,
    Reduce,
    Sort,
    Ternary,
    TopkMoe,
    TurboKv,
    TurboWht,
    Unary,
}

pub const ALL_IDS: [Id; 21] = [
    Id::Affine,
    Id::AttnQ8Kv,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::FlashAttn,
    Id::FlashAttnV2,
    Id::FusedFfnDecode,
    Id::GatedDeltaNet,
    Id::Indexing,
    Id::MmqTurbo,
    Id::Quantized,
    Id::QuantizeKvQ8,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::TopkMoe,
    Id::TurboKv,
    Id::TurboWht,
    Id::Unary,
];

/// A compiled kernel module (HSACO binary).
pub struct Module {
    index: usize,
    hsaco: &'static [u8],
}

impl Module {
    /// Index for caching in `ModuleStore`.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Raw HSACO bytes for `hipModuleLoadData`.
    pub fn hsaco(&self) -> &'static [u8] {
        self.hsaco
    }
}

macro_rules! mdl {
    ($idx:expr, $name:ident) => {
        pub const $name: Module = Module {
            index: $idx,
            hsaco: hsaco::$name,
        };
    };
}

mdl!(0, AFFINE);
mdl!(1, ATTN_Q8_KV);
mdl!(2, BINARY);
mdl!(3, CAST);
mdl!(4, CONV);
mdl!(5, FILL);
mdl!(6, FLASH_ATTN);
mdl!(7, FLASH_ATTN_V2);
mdl!(8, FUSED_FFN_DECODE);
mdl!(9, GATED_DELTA_NET);
mdl!(10, INDEXING);
mdl!(11, MMQ_TURBO);
mdl!(12, QUANTIZED);
mdl!(13, QUANTIZE_KV_Q8);
mdl!(14, REDUCE);
mdl!(15, SORT);
mdl!(16, TERNARY);
mdl!(17, TOPK_MOE);
mdl!(18, TURBO_KV);
mdl!(19, TURBO_WHT);
mdl!(20, UNARY);

/// Produce a kernel function name with dtype suffix, e.g. `"add_f32"`.
pub fn kernel_name<T: crate::DTypeName>(root: &str) -> String {
    format!("{root}_{}", T::NAME)
}

/// Trait for mapping Rust types to kernel dtype name suffixes.
pub trait DTypeName {
    const NAME: &'static str;
}

impl DTypeName for u8 {
    const NAME: &'static str = "u8";
}
impl DTypeName for u32 {
    const NAME: &'static str = "u32";
}
impl DTypeName for i16 {
    const NAME: &'static str = "i16";
}
impl DTypeName for i32 {
    const NAME: &'static str = "i32";
}
impl DTypeName for i64 {
    const NAME: &'static str = "i64";
}
impl DTypeName for half::f16 {
    const NAME: &'static str = "f16";
}
impl DTypeName for half::bf16 {
    const NAME: &'static str = "bf16";
}
impl DTypeName for f32 {
    const NAME: &'static str = "f32";
}
impl DTypeName for f64 {
    const NAME: &'static str = "f64";
}
