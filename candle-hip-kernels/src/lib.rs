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
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 16] = [
    Id::Affine,
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
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
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
mdl!(1, BINARY);
mdl!(2, CAST);
mdl!(3, CONV);
mdl!(4, FILL);
mdl!(5, FLASH_ATTN);
mdl!(6, FLASH_ATTN_V2);
mdl!(7, FUSED_FFN_DECODE);
mdl!(8, GATED_DELTA_NET);
mdl!(9, INDEXING);
mdl!(10, MMQ_TURBO);
mdl!(11, QUANTIZED);
mdl!(12, REDUCE);
mdl!(13, SORT);
mdl!(14, TERNARY);
mdl!(15, UNARY);

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
