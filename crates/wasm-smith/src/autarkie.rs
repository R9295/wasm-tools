use crate::core::Code;
use crate::core::DataSegment;
use crate::core::DuplicateImportsBehavior;
use crate::core::ElementSegment;
use crate::core::FuncType;
use crate::core::GlobalType;
use crate::core::Import;
use crate::core::Instructions;
use crate::core::MaxTypeLimit;
use crate::core::SubType;
use crate::core::TableType;
use crate::core::TagType;
use crate::Config;
use crate::HashSet;
use crate::MemoryType;
use crate::Module;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;
use wasm_encoder::ConstExpr;
use wasm_encoder::ExportKind;
use wasm_encoder::FuzzInstruction;
use wasm_encoder::ValType;

#[derive(Debug, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct FuzzModule {
    config: Config,
    duplicate_imports_behavior: DuplicateImportsBehavior,
    valtypes: Vec<ValType>,
    /// All types locally defined in this module (available in the type index
    /// space).
    types: Vec<SubType>,

    /// Non-overlapping ranges within `types` that belong to the same rec
    /// group. All of `types` is covered by these ranges. When GC is not
    /// enabled, these are all single-element ranges.
    rec_groups: Vec<Range<usize>>,

    /// A map from a super type to all of its sub types.
    super_to_sub_types: HashMap<u32, Vec<u32>>,

    /// Indices within `types` that are not final types.
    can_subtype: Vec<u32>,

    /// Whether we should encode a types section, even if `self.types` is empty.
    should_encode_types: bool,

    /// Whether we should propagate sharedness to types generated inside
    /// `propagate_shared`.
    must_share: bool,
    /// All of this module's imports. These don't have their own index space,
    /// but instead introduce entries to each imported entity's associated index
    /// space.
    imports: Vec<Import>,

    /// Whether we should encode an imports section, even if `self.imports` is
    /// empty.
    should_encode_imports: bool,

    /// Indices within `types` that are array types.
    array_types: Vec<u32>,

    /// Indices within `types` that are function types.
    func_types: Vec<u32>,

    /// Indices within `types that are struct types.
    struct_types: Vec<u32>,

    /// Number of imported items into this module.
    num_imports: usize,

    /// The number of tags defined in this module (not imported or
    /// aliased).
    num_defined_tags: usize,

    /// The number of functions defined in this module (not imported or
    /// aliased).
    num_defined_funcs: usize,

    /// Initialization expressions for all defined tables in this module.
    defined_tables: Vec<Option<ConstExpr>>,

    /// The number of memories defined in this module (not imported or
    /// aliased).
    num_defined_memories: usize,

    /// The indexes and initialization expressions of globals defined in this
    /// module.
    defined_globals: Vec<(u32, ConstExpr)>,

    /// All tags available to this module, sorted by their index. The list
    /// entry is the type of each tag.
    tags: Vec<TagType>,

    /// All functions available to this module, sorted by their index. The list
    /// entry points to the index in this module where the function type is
    /// defined (if available) and provides the type of the function.
    funcs: Vec<(u32, Rc<FuncType>)>,

    /// All tables available to this module, sorted by their index. The list
    /// entry is the type of each table.
    tables: Vec<TableType>,

    /// All globals available to this module, sorted by their index. The list
    /// entry is the type of each global.
    globals: Vec<GlobalType>,

    /// All memories available to this module, sorted by their index. The list
    /// entry is the type of each memory.
    memories: Vec<MemoryType>,

    exports: Vec<(String, ExportKind, u32)>,
    start: Option<u32>,
    elems: Vec<ElementSegment>,

    code: Vec<FuzzCode>,

    data: Vec<DataSegment>,
    /// The predicted size of the effective type of this module, based on this
    /// module's size of the types of imports/exports.
    type_size: u32,

    /// Names currently exported from this module.
    export_names: HashSet<String>,

    // Reusable buffer in `self.arbitrary_const_expr` to amortize the cost of
    // allocation.
    // const_expr_choices: Vec<Box<dyn Fn(&mut Unstructured, ValType) -> Result<ConstExpr>>>,
    /// What the maximum type index that can be referenced is.
    max_type_limit: MaxTypeLimit,

    /// Some known-interesting values, such as powers of two, values just before
    /// or just after a memory size, etc...
    interesting_values32: Vec<u32>,
    interesting_values64: Vec<u64>,
}

#[derive(Debug, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct FuzzCode {
    pub locals: Vec<ValType>,
    pub instructions: Vec<FuzzInstruction>,
}

impl From<FuzzModule> for Module {
    fn from(value: FuzzModule) -> Self {
        let code = value
            .code
            .into_iter()
            .map(|code| Code {
                locals: code.locals,
                instructions: Instructions::Generated(
                    code.instructions
                        .into_iter()
                        .map(|insn| insn.into())
                        .collect::<Vec<_>>(),
                ),
            })
            .collect::<Vec<_>>();
        Self {
            config: value.config,
            duplicate_imports_behavior: value.duplicate_imports_behavior,
            valtypes: value.valtypes,
            types: value.types,
            rec_groups: value.rec_groups,
            super_to_sub_types: value.super_to_sub_types,
            can_subtype: value.can_subtype,
            should_encode_types: value.should_encode_types,
            must_share: value.must_share,
            imports: value.imports,
            should_encode_imports: value.should_encode_imports,
            array_types: value.array_types,
            func_types: value.func_types,
            struct_types: value.struct_types,
            num_imports: value.num_imports,
            num_defined_tags: value.num_defined_tags,
            num_defined_funcs: value.num_defined_funcs,
            defined_tables: value.defined_tables,
            num_defined_memories: value.num_defined_memories,
            defined_globals: value.defined_globals,
            tags: value.tags,
            funcs: value.funcs,
            tables: value.tables,
            globals: value.globals,
            memories: value.memories,
            exports: value.exports,
            start: value.start,
            elems: value.elems,
            code,
            const_expr_choices: vec![],
            data: value.data,
            type_size: value.type_size,
            export_names: value.export_names,
            max_type_limit: value.max_type_limit,
            interesting_values32: value.interesting_values32,
            interesting_values64: value.interesting_values64,
        }
    }
}
