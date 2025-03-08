use crate::{
    encode_section, Encode, HeapType, InstructionSink, RefType, Section, SectionId, ValType,
};
use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

/// An encoder for the code section.
///
/// Code sections are only supported for modules.
///
/// # Example
///
/// ```
/// use wasm_encoder::{
///     CodeSection, Function, FunctionSection, Module,
///     TypeSection, ValType
/// };
///
/// let mut types = TypeSection::new();
/// types.ty().function(vec![], vec![ValType::I32]);
///
/// let mut functions = FunctionSection::new();
/// let type_index = 0;
/// functions.function(type_index);
///
/// let locals = vec![];
/// let mut func = Function::new(locals);
/// func.instructions().i32_const(42);
/// let mut code = CodeSection::new();
/// code.function(&func);
///
/// let mut module = Module::new();
/// module
///     .section(&types)
///     .section(&functions)
///     .section(&code);
///
/// let wasm_bytes = module.finish();
/// ```
#[derive(Clone, Default, Debug)]
pub struct CodeSection {
    bytes: Vec<u8>,
    num_added: u32,
}

impl CodeSection {
    /// Create a new code section encoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of functions in the section.
    pub fn len(&self) -> u32 {
        self.num_added
    }

    /// The number of bytes already added to this section.
    ///
    /// This number doesn't include the vector length that precedes the
    /// code entries, since it has a variable size that isn't known until all
    /// functions are added.
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Determines if the section is empty.
    pub fn is_empty(&self) -> bool {
        self.num_added == 0
    }

    /// Write a function body into this code section.
    pub fn function(&mut self, func: &Function) -> &mut Self {
        func.encode(&mut self.bytes);
        self.num_added += 1;
        self
    }

    /// Add a raw byte slice into this code section as a function body.
    ///
    /// The length prefix of the function body will be automatically prepended,
    /// and should not be included in the raw byte slice.
    ///
    /// # Example
    ///
    /// You can use the `raw` method to copy an already-encoded function body
    /// into a new code section encoder:
    ///
    /// ```
    /// # use wasmparser::{BinaryReader, CodeSectionReader};
    /// //                  id, size, # entries, entry
    /// let code_section = [10, 6,    1,         4, 0, 65, 0, 11];
    ///
    /// // Parse the code section.
    /// let reader = BinaryReader::new(&code_section, 0);
    /// let reader = CodeSectionReader::new(reader).unwrap();
    /// let body = reader.into_iter().next().unwrap().unwrap();
    /// let body_range = body.range();
    ///
    /// // Add the body to a new code section encoder by copying bytes rather
    /// // than re-parsing and re-encoding it.
    /// let mut encoder = wasm_encoder::CodeSection::new();
    /// encoder.raw(&code_section[body_range.start..body_range.end]);
    /// ```
    pub fn raw(&mut self, data: &[u8]) -> &mut Self {
        data.encode(&mut self.bytes);
        self.num_added += 1;
        self
    }
}

impl Encode for CodeSection {
    fn encode(&self, sink: &mut Vec<u8>) {
        encode_section(sink, self.num_added, &self.bytes);
    }
}

impl Section for CodeSection {
    fn id(&self) -> u8 {
        SectionId::Code.into()
    }
}

/// An encoder for a function body within the code section.
///
/// # Example
///
/// ```
/// use wasm_encoder::{CodeSection, Function};
///
/// // Define the function body for:
/// //
/// //     (func (param i32 i32) (result i32)
/// //       local.get 0
/// //       local.get 1
/// //       i32.add)
/// let locals = vec![];
/// let mut func = Function::new(locals);
/// func.instructions()
///     .local_get(0)
///     .local_get(1)
///     .i32_add();
///
/// // Add our function to the code section.
/// let mut code = CodeSection::new();
/// code.function(&func);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Function {
    bytes: Vec<u8>,
}

impl Function {
    /// Create a new function body with the given locals.
    ///
    /// The argument is an iterator over `(N, Ty)`, which defines
    /// that the next `N` locals will be of type `Ty`.
    ///
    /// For example, a function with locals 0 and 1 of type I32 and
    /// local 2 of type F32 would be created as:
    ///
    /// ```
    /// # use wasm_encoder::{Function, ValType};
    /// let f = Function::new([(2, ValType::I32), (1, ValType::F32)]);
    /// ```
    ///
    /// For more information about the code section (and function definition) in the WASM binary format
    /// see the [WebAssembly spec](https://webassembly.github.io/spec/core/binary/modules.html#binary-func)
    pub fn new<L>(locals: L) -> Self
    where
        L: IntoIterator<Item = (u32, ValType)>,
        L::IntoIter: ExactSizeIterator,
    {
        let locals = locals.into_iter();
        let mut bytes = vec![];
        locals.len().encode(&mut bytes);
        for (count, ty) in locals {
            count.encode(&mut bytes);
            ty.encode(&mut bytes);
        }
        Function { bytes }
    }

    /// Create a function from a list of locals' types.
    ///
    /// Unlike [`Function::new`], this constructor simply takes a list of types
    /// which are in order associated with locals.
    ///
    /// For example:
    ///
    ///  ```
    /// # use wasm_encoder::{Function, ValType};
    /// let f = Function::new([(2, ValType::I32), (1, ValType::F32)]);
    /// let g = Function::new_with_locals_types([
    ///     ValType::I32, ValType::I32, ValType::F32
    /// ]);
    ///
    /// assert_eq!(f, g)
    /// ```
    pub fn new_with_locals_types<L>(locals: L) -> Self
    where
        L: IntoIterator<Item = ValType>,
    {
        let locals = locals.into_iter();

        let mut locals_collected: Vec<(u32, ValType)> = vec![];
        for l in locals {
            if let Some((last_count, last_type)) = locals_collected.last_mut() {
                if l == *last_type {
                    // Increment the count of consecutive locals of this type
                    *last_count += 1;
                    continue;
                }
            }
            // If we didn't increment, a new type of local appeared
            locals_collected.push((1, l));
        }

        Function::new(locals_collected)
    }

    /// Get an instruction encoder for this function body.
    pub fn instructions(&mut self) -> InstructionSink {
        InstructionSink::new(&mut self.bytes)
    }

    /// Write an instruction into this function body.
    pub fn instruction(&mut self, instruction: &Instruction) -> &mut Self {
        instruction.encode(&mut self.bytes);
        self
    }

    /// Add raw bytes to this function's body.
    pub fn raw<B>(&mut self, bytes: B) -> &mut Self
    where
        B: IntoIterator<Item = u8>,
    {
        self.bytes.extend(bytes);
        self
    }

    /// The number of bytes already added to this function.
    ///
    /// This number doesn't include the variable-width size field that `encode`
    /// will write before the added bytes, since the size of that field isn't
    /// known until all the instructions are added to this function.
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Unwraps and returns the raw byte encoding of this function.
    ///
    /// This encoding doesn't include the variable-width size field
    /// that `encode` will write before the added bytes. As such, its
    /// length will match the return value of [`Function::byte_len`].
    ///
    /// # Use Case
    ///
    /// This raw byte form is suitable for later using with
    /// [`CodeSection::raw`]. Note that it *differs* from what results
    /// from [`Function::encode`] precisely due to the *lack* of the
    /// length prefix; [`CodeSection::raw`] will use this. Using
    /// [`Function::encode`] instead produces bytes that cannot be fed
    /// into other wasm-encoder types without stripping off the length
    /// prefix, which is awkward and error-prone.
    ///
    /// This method combined with [`CodeSection::raw`] may be useful
    /// together if one wants to save the result of function encoding
    /// and use it later: for example, caching the result of some code
    /// generation process.
    ///
    /// For example:
    ///
    /// ```
    /// # use wasm_encoder::{CodeSection, Function};
    /// let mut f = Function::new([]);
    /// f.instructions().end();
    /// let bytes = f.into_raw_body();
    /// // (save `bytes` somewhere for later use)
    /// let mut code = CodeSection::new();
    /// code.raw(&bytes[..]);
    ///
    /// assert_eq!(2, bytes.len());  // Locals count, then `end`
    /// assert_eq!(3, code.byte_len()); // Function length byte, function body
    /// ```
    pub fn into_raw_body(self) -> Vec<u8> {
        self.bytes
    }
}

impl Encode for Function {
    fn encode(&self, sink: &mut Vec<u8>) {
        self.bytes.encode(sink);
    }
}

/// The immediate for a memory instruction.
#[derive(Clone, Copy, Debug, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct MemArg {
    /// A static offset to add to the instruction's dynamic address operand.
    ///
    /// This is a `u64` field for the memory64 proposal, but 32-bit memories
    /// limit offsets to at most `u32::MAX` bytes. This will be encoded as a LEB
    /// but it won't generate a valid module if an offset is specified which is
    /// larger than the maximum size of the index space for the memory indicated
    /// by `memory_index`.
    pub offset: u64,
    /// The expected alignment of the instruction's dynamic address operand
    /// (expressed the exponent of a power of two).
    pub align: u32,
    /// The index of the memory this instruction is operating upon.
    pub memory_index: u32,
}

impl Encode for MemArg {
    fn encode(&self, sink: &mut Vec<u8>) {
        if self.memory_index == 0 {
            self.align.encode(sink);
            self.offset.encode(sink);
        } else {
            (self.align | (1 << 6)).encode(sink);
            self.memory_index.encode(sink);
            self.offset.encode(sink);
        }
    }
}

/// The memory ordering for atomic instructions.
///
/// For an in-depth explanation of memory orderings, see the C++ documentation
/// for [`memory_order`] or the Rust documentation for [`atomic::Ordering`].
///
/// [`memory_order`]: https://en.cppreference.com/w/cpp/atomic/memory_order
/// [`atomic::Ordering`]: https://doc.rust-lang.org/std/sync/atomic/enum.Ordering.html
#[derive(Clone, Copy, Debug, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum Ordering {
    /// For a load, it acquires; this orders all operations before the last
    /// "releasing" store. For a store, it releases; this orders all operations
    /// before it at the next "acquiring" load.
    AcqRel,
    /// Like `AcqRel` but all threads see all sequentially consistent operations
    /// in the same order.
    SeqCst,
}

impl Encode for Ordering {
    fn encode(&self, sink: &mut Vec<u8>) {
        let flag: u8 = match self {
            Ordering::SeqCst => 0,
            Ordering::AcqRel => 1,
        };
        sink.push(flag);
    }
}

/// Describe an unchecked SIMD lane index.
pub type Lane = u8;

/// The type for a `block`/`if`/`loop`.
#[derive(Clone, Copy, Debug, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum BlockType {
    /// `[] -> []`
    Empty,
    /// `[] -> [t]`
    Result(ValType),
    /// The `n`th function type.
    FunctionType(u32),
}

impl Encode for BlockType {
    fn encode(&self, sink: &mut Vec<u8>) {
        match *self {
            Self::Empty => sink.push(0x40),
            Self::Result(ty) => ty.encode(sink),
            Self::FunctionType(f) => (f as i64).encode(sink),
        }
    }
}

impl From<FuzzInstruction> for Instruction<'_> {
    fn from(instruction: FuzzInstruction) -> Self {
        match instruction {
            // Control instructions
            FuzzInstruction::Unreachable => Instruction::Unreachable,
            FuzzInstruction::Nop => Instruction::Nop,
            FuzzInstruction::Block(block_type) => Instruction::Block(block_type),
            FuzzInstruction::Loop(block_type) => Instruction::Loop(block_type),
            FuzzInstruction::If(block_type) => Instruction::If(block_type),
            FuzzInstruction::Else => Instruction::Else,
            FuzzInstruction::End => Instruction::End,
            FuzzInstruction::Br(label_idx) => Instruction::Br(label_idx),
            FuzzInstruction::BrIf(label_idx) => Instruction::BrIf(label_idx),
            FuzzInstruction::BrTable(labels, default) => {
                Instruction::BrTable(labels.into(), default)
            }
            FuzzInstruction::BrOnNull(label_idx) => Instruction::BrOnNull(label_idx),
            FuzzInstruction::BrOnNonNull(label_idx) => Instruction::BrOnNonNull(label_idx),
            FuzzInstruction::Return => Instruction::Return,
            FuzzInstruction::Call(func_idx) => Instruction::Call(func_idx),
            FuzzInstruction::CallRef(type_idx) => Instruction::CallRef(type_idx),
            FuzzInstruction::CallIndirect {
                type_index,
                table_index,
            } => Instruction::CallIndirect {
                type_index,
                table_index,
            },
            FuzzInstruction::ReturnCallRef(type_idx) => Instruction::ReturnCallRef(type_idx),
            FuzzInstruction::ReturnCall(func_idx) => Instruction::ReturnCall(func_idx),
            FuzzInstruction::ReturnCallIndirect {
                type_index,
                table_index,
            } => Instruction::ReturnCallIndirect {
                type_index,
                table_index,
            },
            FuzzInstruction::TryTable(block_type, catches) => {
                Instruction::TryTable(block_type, catches.into())
            }
            FuzzInstruction::Throw(tag) => Instruction::Throw(tag),
            FuzzInstruction::ThrowRef => Instruction::ThrowRef,

            // Deprecated exception-handling instructions
            FuzzInstruction::Try(block_type) => Instruction::Try(block_type),
            FuzzInstruction::Delegate(label_idx) => Instruction::Delegate(label_idx),
            FuzzInstruction::Catch(tag) => Instruction::Catch(tag),
            FuzzInstruction::CatchAll => Instruction::CatchAll,
            FuzzInstruction::Rethrow(label_idx) => Instruction::Rethrow(label_idx),

            // Parametric instructions
            FuzzInstruction::Drop => Instruction::Drop,
            FuzzInstruction::Select => Instruction::Select,

            // Variable instructions
            FuzzInstruction::LocalGet(local_idx) => Instruction::LocalGet(local_idx),
            FuzzInstruction::LocalSet(local_idx) => Instruction::LocalSet(local_idx),
            FuzzInstruction::LocalTee(local_idx) => Instruction::LocalTee(local_idx),
            FuzzInstruction::GlobalGet(global_idx) => Instruction::GlobalGet(global_idx),
            FuzzInstruction::GlobalSet(global_idx) => Instruction::GlobalSet(global_idx),

            // Memory instructions
            FuzzInstruction::I32Load(mem_arg) => Instruction::I32Load(mem_arg),
            FuzzInstruction::I64Load(mem_arg) => Instruction::I64Load(mem_arg),
            FuzzInstruction::F32Load(mem_arg) => Instruction::F32Load(mem_arg),
            FuzzInstruction::F64Load(mem_arg) => Instruction::F64Load(mem_arg),
            FuzzInstruction::I32Load8S(mem_arg) => Instruction::I32Load8S(mem_arg),
            FuzzInstruction::I32Load8U(mem_arg) => Instruction::I32Load8U(mem_arg),
            FuzzInstruction::I32Load16S(mem_arg) => Instruction::I32Load16S(mem_arg),
            FuzzInstruction::I32Load16U(mem_arg) => Instruction::I32Load16U(mem_arg),
            FuzzInstruction::I64Load8S(mem_arg) => Instruction::I64Load8S(mem_arg),
            FuzzInstruction::I64Load8U(mem_arg) => Instruction::I64Load8U(mem_arg),
            FuzzInstruction::I64Load16S(mem_arg) => Instruction::I64Load16S(mem_arg),
            FuzzInstruction::I64Load16U(mem_arg) => Instruction::I64Load16U(mem_arg),
            FuzzInstruction::I64Load32S(mem_arg) => Instruction::I64Load32S(mem_arg),
            FuzzInstruction::I64Load32U(mem_arg) => Instruction::I64Load32U(mem_arg),
            FuzzInstruction::I32Store(mem_arg) => Instruction::I32Store(mem_arg),
            FuzzInstruction::I64Store(mem_arg) => Instruction::I64Store(mem_arg),
            FuzzInstruction::F32Store(mem_arg) => Instruction::F32Store(mem_arg),
            FuzzInstruction::F64Store(mem_arg) => Instruction::F64Store(mem_arg),
            FuzzInstruction::I32Store8(mem_arg) => Instruction::I32Store8(mem_arg),
            FuzzInstruction::I32Store16(mem_arg) => Instruction::I32Store16(mem_arg),
            FuzzInstruction::I64Store8(mem_arg) => Instruction::I64Store8(mem_arg),
            FuzzInstruction::I64Store16(mem_arg) => Instruction::I64Store16(mem_arg),
            FuzzInstruction::I64Store32(mem_arg) => Instruction::I64Store32(mem_arg),
            FuzzInstruction::MemorySize(mem_idx) => Instruction::MemorySize(mem_idx),
            FuzzInstruction::MemoryGrow(mem_idx) => Instruction::MemoryGrow(mem_idx),
            FuzzInstruction::MemoryInit { mem, data_index } => {
                Instruction::MemoryInit { mem, data_index }
            }
            // Additional memory instructions
            FuzzInstruction::DataDrop(data_idx) => Instruction::DataDrop(data_idx),
            FuzzInstruction::MemoryCopy { src_mem, dst_mem } => {
                Instruction::MemoryCopy { src_mem, dst_mem }
            }
            FuzzInstruction::MemoryFill(mem_idx) => Instruction::MemoryFill(mem_idx),
            FuzzInstruction::MemoryDiscard(mem_idx) => Instruction::MemoryDiscard(mem_idx),

            // Numeric instructions
            FuzzInstruction::I32Const(val) => Instruction::I32Const(val),
            FuzzInstruction::I64Const(val) => Instruction::I64Const(val),
            FuzzInstruction::F32Const(val) => Instruction::F32Const(val),
            FuzzInstruction::F64Const(val) => Instruction::F64Const(val),
            FuzzInstruction::I32Eqz => Instruction::I32Eqz,
            FuzzInstruction::I32Eq => Instruction::I32Eq,
            FuzzInstruction::I32Ne => Instruction::I32Ne,
            FuzzInstruction::I32LtS => Instruction::I32LtS,
            FuzzInstruction::I32LtU => Instruction::I32LtU,
            FuzzInstruction::I32GtS => Instruction::I32GtS,
            FuzzInstruction::I32GtU => Instruction::I32GtU,
            FuzzInstruction::I32LeS => Instruction::I32LeS,
            FuzzInstruction::I32LeU => Instruction::I32LeU,
            FuzzInstruction::I32GeS => Instruction::I32GeS,
            FuzzInstruction::I32GeU => Instruction::I32GeU,
            FuzzInstruction::I64Eqz => Instruction::I64Eqz,
            FuzzInstruction::I64Eq => Instruction::I64Eq,
            FuzzInstruction::I64Ne => Instruction::I64Ne,
            FuzzInstruction::I64LtS => Instruction::I64LtS,
            FuzzInstruction::I64LtU => Instruction::I64LtU,
            FuzzInstruction::I64GtS => Instruction::I64GtS,
            FuzzInstruction::I64GtU => Instruction::I64GtU,
            FuzzInstruction::I64LeS => Instruction::I64LeS,
            FuzzInstruction::I64LeU => Instruction::I64LeU,
            FuzzInstruction::I64GeS => Instruction::I64GeS,
            FuzzInstruction::I64GeU => Instruction::I64GeU,
            FuzzInstruction::F32Eq => Instruction::F32Eq,
            FuzzInstruction::F32Ne => Instruction::F32Ne,
            FuzzInstruction::F32Lt => Instruction::F32Lt,
            FuzzInstruction::F32Gt => Instruction::F32Gt,
            FuzzInstruction::F32Le => Instruction::F32Le,
            FuzzInstruction::F32Ge => Instruction::F32Ge,
            FuzzInstruction::F64Eq => Instruction::F64Eq,
            FuzzInstruction::F64Ne => Instruction::F64Ne,
            FuzzInstruction::F64Lt => Instruction::F64Lt,
            FuzzInstruction::F64Gt => Instruction::F64Gt,
            FuzzInstruction::F64Le => Instruction::F64Le,
            FuzzInstruction::F64Ge => Instruction::F64Ge,
            FuzzInstruction::I32Clz => Instruction::I32Clz,
            FuzzInstruction::I32Ctz => Instruction::I32Ctz,
            FuzzInstruction::I32Popcnt => Instruction::I32Popcnt,
            FuzzInstruction::I32Add => Instruction::I32Add,
            FuzzInstruction::I32Sub => Instruction::I32Sub,
            FuzzInstruction::I32Mul => Instruction::I32Mul,
            FuzzInstruction::I32DivS => Instruction::I32DivS,
            FuzzInstruction::I32DivU => Instruction::I32DivU,
            FuzzInstruction::I32RemS => Instruction::I32RemS,
            FuzzInstruction::I32RemU => Instruction::I32RemU,
            FuzzInstruction::I32And => Instruction::I32And,
            FuzzInstruction::I32Or => Instruction::I32Or,
            FuzzInstruction::I32Xor => Instruction::I32Xor,
            FuzzInstruction::I32Shl => Instruction::I32Shl,
            FuzzInstruction::I32ShrS => Instruction::I32ShrS,
            FuzzInstruction::I32ShrU => Instruction::I32ShrU,
            FuzzInstruction::I32Rotl => Instruction::I32Rotl,
            FuzzInstruction::I32Rotr => Instruction::I32Rotr,
            FuzzInstruction::I64Clz => Instruction::I64Clz,
            FuzzInstruction::I64Ctz => Instruction::I64Ctz,
            FuzzInstruction::I64Popcnt => Instruction::I64Popcnt,
            FuzzInstruction::I64Add => Instruction::I64Add,
            FuzzInstruction::I64Sub => Instruction::I64Sub,
            FuzzInstruction::I64Mul => Instruction::I64Mul,
            FuzzInstruction::I64DivS => Instruction::I64DivS,
            FuzzInstruction::I64DivU => Instruction::I64DivU,
            FuzzInstruction::I64RemS => Instruction::I64RemS,
            FuzzInstruction::I64RemU => Instruction::I64RemU,
            FuzzInstruction::I64And => Instruction::I64And,
            FuzzInstruction::I64Or => Instruction::I64Or,
            FuzzInstruction::I64Xor => Instruction::I64Xor,
            FuzzInstruction::I64Shl => Instruction::I64Shl,
            FuzzInstruction::I64ShrS => Instruction::I64ShrS,
            FuzzInstruction::I64ShrU => Instruction::I64ShrU,
            FuzzInstruction::I64Rotl => Instruction::I64Rotl,
            FuzzInstruction::I64Rotr => Instruction::I64Rotr,
            FuzzInstruction::F32Abs => Instruction::F32Abs,
            FuzzInstruction::F32Neg => Instruction::F32Neg,
            FuzzInstruction::F32Ceil => Instruction::F32Ceil,
            FuzzInstruction::F32Floor => Instruction::F32Floor,
            FuzzInstruction::F32Trunc => Instruction::F32Trunc,
            FuzzInstruction::F32Nearest => Instruction::F32Nearest,
            FuzzInstruction::F32Sqrt => Instruction::F32Sqrt,
            FuzzInstruction::F32Add => Instruction::F32Add,
            FuzzInstruction::F32Sub => Instruction::F32Sub,
            FuzzInstruction::F32Mul => Instruction::F32Mul,
            FuzzInstruction::F32Div => Instruction::F32Div,
            FuzzInstruction::F32Min => Instruction::F32Min,
            FuzzInstruction::F32Max => Instruction::F32Max,
            FuzzInstruction::F32Copysign => Instruction::F32Copysign,
            FuzzInstruction::F64Abs => Instruction::F64Abs,
            // F64 floating point operations
            FuzzInstruction::F64Neg => Instruction::F64Neg,
            FuzzInstruction::F64Ceil => Instruction::F64Ceil,
            FuzzInstruction::F64Floor => Instruction::F64Floor,
            FuzzInstruction::F64Trunc => Instruction::F64Trunc,
            FuzzInstruction::F64Nearest => Instruction::F64Nearest,
            FuzzInstruction::F64Sqrt => Instruction::F64Sqrt,
            FuzzInstruction::F64Add => Instruction::F64Add,
            FuzzInstruction::F64Sub => Instruction::F64Sub,
            FuzzInstruction::F64Mul => Instruction::F64Mul,
            FuzzInstruction::F64Div => Instruction::F64Div,
            FuzzInstruction::F64Min => Instruction::F64Min,
            FuzzInstruction::F64Max => Instruction::F64Max,
            FuzzInstruction::F64Copysign => Instruction::F64Copysign,

            // Conversion operations
            FuzzInstruction::I32WrapI64 => Instruction::I32WrapI64,
            FuzzInstruction::I32TruncF32S => Instruction::I32TruncF32S,
            FuzzInstruction::I32TruncF32U => Instruction::I32TruncF32U,
            FuzzInstruction::I32TruncF64S => Instruction::I32TruncF64S,
            FuzzInstruction::I32TruncF64U => Instruction::I32TruncF64U,
            FuzzInstruction::I64ExtendI32S => Instruction::I64ExtendI32S,
            FuzzInstruction::I64ExtendI32U => Instruction::I64ExtendI32U,
            FuzzInstruction::I64TruncF32S => Instruction::I64TruncF32S,
            FuzzInstruction::I64TruncF32U => Instruction::I64TruncF32U,
            FuzzInstruction::I64TruncF64S => Instruction::I64TruncF64S,
            FuzzInstruction::I64TruncF64U => Instruction::I64TruncF64U,
            FuzzInstruction::F32ConvertI32S => Instruction::F32ConvertI32S,
            FuzzInstruction::F32ConvertI32U => Instruction::F32ConvertI32U,
            FuzzInstruction::F32ConvertI64S => Instruction::F32ConvertI64S,
            FuzzInstruction::F32ConvertI64U => Instruction::F32ConvertI64U,
            FuzzInstruction::F32DemoteF64 => Instruction::F32DemoteF64,
            FuzzInstruction::F64ConvertI32S => Instruction::F64ConvertI32S,
            FuzzInstruction::F64ConvertI32U => Instruction::F64ConvertI32U,
            FuzzInstruction::F64ConvertI64S => Instruction::F64ConvertI64S,
            FuzzInstruction::F64ConvertI64U => Instruction::F64ConvertI64U,
            FuzzInstruction::F64PromoteF32 => Instruction::F64PromoteF32,
            FuzzInstruction::I32ReinterpretF32 => Instruction::I32ReinterpretF32,
            FuzzInstruction::I64ReinterpretF64 => Instruction::I64ReinterpretF64,
            FuzzInstruction::F32ReinterpretI32 => Instruction::F32ReinterpretI32,
            FuzzInstruction::F64ReinterpretI64 => Instruction::F64ReinterpretI64,

            // Sign extension operations
            FuzzInstruction::I32Extend8S => Instruction::I32Extend8S,
            FuzzInstruction::I32Extend16S => Instruction::I32Extend16S,
            FuzzInstruction::I64Extend8S => Instruction::I64Extend8S,
            FuzzInstruction::I64Extend16S => Instruction::I64Extend16S,
            FuzzInstruction::I64Extend32S => Instruction::I64Extend32S,

            // Saturating truncation operations
            FuzzInstruction::I32TruncSatF32S => Instruction::I32TruncSatF32S,
            FuzzInstruction::I32TruncSatF32U => Instruction::I32TruncSatF32U,
            FuzzInstruction::I32TruncSatF64S => Instruction::I32TruncSatF64S,
            FuzzInstruction::I32TruncSatF64U => Instruction::I32TruncSatF64U,
            FuzzInstruction::I64TruncSatF32S => Instruction::I64TruncSatF32S,
            FuzzInstruction::I64TruncSatF32U => Instruction::I64TruncSatF32U,
            FuzzInstruction::I64TruncSatF64S => Instruction::I64TruncSatF64S,
            FuzzInstruction::I64TruncSatF64U => Instruction::I64TruncSatF64U,

            // Reference types instructions
            FuzzInstruction::TypedSelect(val_type) => Instruction::TypedSelect(val_type),
            FuzzInstruction::RefNull(heap_type) => Instruction::RefNull(heap_type),
            FuzzInstruction::RefIsNull => Instruction::RefIsNull,
            FuzzInstruction::RefFunc(func_idx) => Instruction::RefFunc(func_idx),
            FuzzInstruction::RefEq => Instruction::RefEq,
            FuzzInstruction::RefAsNonNull => Instruction::RefAsNonNull,

            // GC types instructions
            FuzzInstruction::StructNew(type_idx) => Instruction::StructNew(type_idx),
            FuzzInstruction::StructNewDefault(type_idx) => Instruction::StructNewDefault(type_idx),
            FuzzInstruction::StructGet {
                struct_type_index,
                field_index,
            } => Instruction::StructGet {
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructGetS {
                struct_type_index,
                field_index,
            } => Instruction::StructGetS {
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructGetU {
                struct_type_index,
                field_index,
            } => Instruction::StructGetU {
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructSet {
                struct_type_index,
                field_index,
            } => Instruction::StructSet {
                struct_type_index,
                field_index,
            },
            FuzzInstruction::ArrayNew(type_idx) => Instruction::ArrayNew(type_idx),
            FuzzInstruction::ArrayNewDefault(type_idx) => Instruction::ArrayNewDefault(type_idx),
            FuzzInstruction::ArrayNewFixed {
                array_type_index,
                array_size,
            } => Instruction::ArrayNewFixed {
                array_type_index,
                array_size,
            },
            FuzzInstruction::ArrayNewData {
                array_type_index,
                array_data_index,
            } => Instruction::ArrayNewData {
                array_type_index,
                array_data_index,
            },
            FuzzInstruction::ArrayNewElem {
                array_type_index,
                array_elem_index,
            } => Instruction::ArrayNewElem {
                array_type_index,
                array_elem_index,
            },
            FuzzInstruction::ArrayGet(type_idx) => Instruction::ArrayGet(type_idx),
            FuzzInstruction::ArrayGetS(type_idx) => Instruction::ArrayGetS(type_idx),
            FuzzInstruction::ArrayGetU(type_idx) => Instruction::ArrayGetU(type_idx),
            FuzzInstruction::ArraySet(type_idx) => Instruction::ArraySet(type_idx),
            FuzzInstruction::ArrayLen => Instruction::ArrayLen,
            FuzzInstruction::ArrayFill(type_idx) => Instruction::ArrayFill(type_idx),
            FuzzInstruction::ArrayCopy {
                array_type_index_dst,
                array_type_index_src,
            } => Instruction::ArrayCopy {
                array_type_index_dst,
                array_type_index_src,
            },
            FuzzInstruction::ArrayInitData {
                array_type_index,
                array_data_index,
            } => Instruction::ArrayInitData {
                array_type_index,
                array_data_index,
            },
            FuzzInstruction::ArrayInitElem {
                array_type_index,
                array_elem_index,
            } => Instruction::ArrayInitElem {
                array_type_index,
                array_elem_index,
            },
            // Reference type operations
            FuzzInstruction::RefTestNonNull(heap_type) => Instruction::RefTestNonNull(heap_type),
            FuzzInstruction::RefTestNullable(heap_type) => Instruction::RefTestNullable(heap_type),
            FuzzInstruction::RefCastNonNull(heap_type) => Instruction::RefCastNonNull(heap_type),
            FuzzInstruction::RefCastNullable(heap_type) => Instruction::RefCastNullable(heap_type),
            FuzzInstruction::BrOnCast {
                relative_depth,
                from_ref_type,
                to_ref_type,
            } => Instruction::BrOnCast {
                relative_depth,
                from_ref_type,
                to_ref_type,
            },
            FuzzInstruction::BrOnCastFail {
                relative_depth,
                from_ref_type,
                to_ref_type,
            } => Instruction::BrOnCastFail {
                relative_depth,
                from_ref_type,
                to_ref_type,
            },
            FuzzInstruction::AnyConvertExtern => Instruction::AnyConvertExtern,
            FuzzInstruction::ExternConvertAny => Instruction::ExternConvertAny,
            FuzzInstruction::RefI31 => Instruction::RefI31,
            FuzzInstruction::I31GetS => Instruction::I31GetS,
            FuzzInstruction::I31GetU => Instruction::I31GetU,

            // Bulk memory instructions
            FuzzInstruction::TableInit { elem_index, table } => {
                Instruction::TableInit { elem_index, table }
            }
            FuzzInstruction::ElemDrop(index) => Instruction::ElemDrop(index),
            FuzzInstruction::TableFill(table_idx) => Instruction::TableFill(table_idx),
            FuzzInstruction::TableSet(table_idx) => Instruction::TableSet(table_idx),
            FuzzInstruction::TableGet(table_idx) => Instruction::TableGet(table_idx),
            FuzzInstruction::TableGrow(table_idx) => Instruction::TableGrow(table_idx),
            FuzzInstruction::TableSize(table_idx) => Instruction::TableSize(table_idx),
            FuzzInstruction::TableCopy {
                src_table,
                dst_table,
            } => Instruction::TableCopy {
                src_table,
                dst_table,
            },

            // SIMD instructions
            FuzzInstruction::V128Load(mem_arg) => Instruction::V128Load(mem_arg),
            FuzzInstruction::V128Load8x8S(mem_arg) => Instruction::V128Load8x8S(mem_arg),
            FuzzInstruction::V128Load8x8U(mem_arg) => Instruction::V128Load8x8U(mem_arg),
            FuzzInstruction::V128Load16x4S(mem_arg) => Instruction::V128Load16x4S(mem_arg),
            FuzzInstruction::V128Load16x4U(mem_arg) => Instruction::V128Load16x4U(mem_arg),
            FuzzInstruction::V128Load32x2S(mem_arg) => Instruction::V128Load32x2S(mem_arg),
            FuzzInstruction::V128Load32x2U(mem_arg) => Instruction::V128Load32x2U(mem_arg),
            FuzzInstruction::V128Load8Splat(mem_arg) => Instruction::V128Load8Splat(mem_arg),
            FuzzInstruction::V128Load16Splat(mem_arg) => Instruction::V128Load16Splat(mem_arg),
            FuzzInstruction::V128Load32Splat(mem_arg) => Instruction::V128Load32Splat(mem_arg),
            FuzzInstruction::V128Load64Splat(mem_arg) => Instruction::V128Load64Splat(mem_arg),
            FuzzInstruction::V128Load32Zero(mem_arg) => Instruction::V128Load32Zero(mem_arg),
            FuzzInstruction::V128Load64Zero(mem_arg) => Instruction::V128Load64Zero(mem_arg),
            FuzzInstruction::V128Store(mem_arg) => Instruction::V128Store(mem_arg),
            FuzzInstruction::V128Load8Lane { memarg, lane } => {
                Instruction::V128Load8Lane { memarg, lane }
            }
            FuzzInstruction::V128Load16Lane { memarg, lane } => {
                Instruction::V128Load16Lane { memarg, lane }
            }
            FuzzInstruction::V128Load32Lane { memarg, lane } => {
                Instruction::V128Load32Lane { memarg, lane }
            }
            FuzzInstruction::V128Load64Lane { memarg, lane } => {
                Instruction::V128Load64Lane { memarg, lane }
            }
            FuzzInstruction::V128Store8Lane { memarg, lane } => {
                Instruction::V128Store8Lane { memarg, lane }
            }
            FuzzInstruction::V128Store16Lane { memarg, lane } => {
                Instruction::V128Store16Lane { memarg, lane }
            }
            FuzzInstruction::V128Store32Lane { memarg, lane } => {
                Instruction::V128Store32Lane { memarg, lane }
            }
            FuzzInstruction::V128Store64Lane { memarg, lane } => {
                Instruction::V128Store64Lane { memarg, lane }
            }
            FuzzInstruction::V128Const(val) => Instruction::V128Const(val),
            FuzzInstruction::I8x16Shuffle(lanes) => Instruction::I8x16Shuffle(lanes),
            FuzzInstruction::I8x16ExtractLaneS(lane) => Instruction::I8x16ExtractLaneS(lane),
            FuzzInstruction::I8x16ExtractLaneU(lane) => Instruction::I8x16ExtractLaneU(lane),
            FuzzInstruction::I8x16ReplaceLane(lane) => Instruction::I8x16ReplaceLane(lane),
            FuzzInstruction::I16x8ExtractLaneS(lane) => Instruction::I16x8ExtractLaneS(lane),
            FuzzInstruction::I16x8ExtractLaneU(lane) => Instruction::I16x8ExtractLaneU(lane),
            FuzzInstruction::I16x8ReplaceLane(lane) => Instruction::I16x8ReplaceLane(lane),
            FuzzInstruction::I32x4ExtractLane(lane) => Instruction::I32x4ExtractLane(lane),
            FuzzInstruction::I32x4ReplaceLane(lane) => Instruction::I32x4ReplaceLane(lane),
            FuzzInstruction::I64x2ExtractLane(lane) => Instruction::I64x2ExtractLane(lane),
            FuzzInstruction::I64x2ReplaceLane(lane) => Instruction::I64x2ReplaceLane(lane),
            FuzzInstruction::F32x4ExtractLane(lane) => Instruction::F32x4ExtractLane(lane),
            FuzzInstruction::F32x4ReplaceLane(lane) => Instruction::F32x4ReplaceLane(lane),
            FuzzInstruction::F64x2ExtractLane(lane) => Instruction::F64x2ExtractLane(lane),
            FuzzInstruction::F64x2ReplaceLane(lane) => Instruction::F64x2ReplaceLane(lane),
            FuzzInstruction::I8x16Swizzle => Instruction::I8x16Swizzle,
            FuzzInstruction::I8x16Splat => Instruction::I8x16Splat,
            FuzzInstruction::I16x8Splat => Instruction::I16x8Splat,
            FuzzInstruction::I32x4Splat => Instruction::I32x4Splat,
            FuzzInstruction::I64x2Splat => Instruction::I64x2Splat,
            FuzzInstruction::F32x4Splat => Instruction::F32x4Splat,
            FuzzInstruction::F64x2Splat => Instruction::F64x2Splat,
            FuzzInstruction::I8x16Eq => Instruction::I8x16Eq,
            FuzzInstruction::I8x16Ne => Instruction::I8x16Ne,
            FuzzInstruction::I8x16LtS => Instruction::I8x16LtS,
            FuzzInstruction::I8x16LtU => Instruction::I8x16LtU,
            FuzzInstruction::I8x16GtS => Instruction::I8x16GtS,
            FuzzInstruction::I8x16GtU => Instruction::I8x16GtU,
            FuzzInstruction::I8x16LeS => Instruction::I8x16LeS,
            FuzzInstruction::I8x16LeU => Instruction::I8x16LeU,
            FuzzInstruction::I8x16GeS => Instruction::I8x16GeS,
            FuzzInstruction::I8x16GeU => Instruction::I8x16GeU,
            FuzzInstruction::I16x8Eq => Instruction::I16x8Eq,
            FuzzInstruction::I16x8Ne => Instruction::I16x8Ne,
            FuzzInstruction::I16x8LtS => Instruction::I16x8LtS,
            FuzzInstruction::I16x8LtU => Instruction::I16x8LtU,
            FuzzInstruction::I16x8GtS => Instruction::I16x8GtS,
            FuzzInstruction::I16x8GtU => Instruction::I16x8GtU,
            FuzzInstruction::I16x8LeS => Instruction::I16x8LeS,
            FuzzInstruction::I16x8LeU => Instruction::I16x8LeU,
            FuzzInstruction::I16x8GeS => Instruction::I16x8GeS,
            FuzzInstruction::I16x8GeU => Instruction::I16x8GeU,
            FuzzInstruction::I32x4Eq => Instruction::I32x4Eq,
            FuzzInstruction::I32x4Ne => Instruction::I32x4Ne,
            FuzzInstruction::I32x4LtS => Instruction::I32x4LtS,
            FuzzInstruction::I32x4LtU => Instruction::I32x4LtU,
            FuzzInstruction::I32x4GtS => Instruction::I32x4GtS,
            FuzzInstruction::I32x4GtU => Instruction::I32x4GtU,
            FuzzInstruction::I32x4LeS => Instruction::I32x4LeS,
            FuzzInstruction::I32x4LeU => Instruction::I32x4LeU,
            FuzzInstruction::I32x4GeS => Instruction::I32x4GeS,
            FuzzInstruction::I32x4GeU => Instruction::I32x4GeU,
            FuzzInstruction::I64x2Eq => Instruction::I64x2Eq,
            FuzzInstruction::I64x2Ne => Instruction::I64x2Ne,
            FuzzInstruction::I64x2LtS => Instruction::I64x2LtS,
            FuzzInstruction::I64x2GtS => Instruction::I64x2GtS,
            FuzzInstruction::I64x2LeS => Instruction::I64x2LeS,
            FuzzInstruction::I64x2GeS => Instruction::I64x2GeS,
            FuzzInstruction::F32x4Eq => Instruction::F32x4Eq,
            FuzzInstruction::F32x4Ne => Instruction::F32x4Ne,
            FuzzInstruction::F32x4Lt => Instruction::F32x4Lt,
            FuzzInstruction::F32x4Gt => Instruction::F32x4Gt,
            FuzzInstruction::F32x4Le => Instruction::F32x4Le,
            FuzzInstruction::F32x4Ge => Instruction::F32x4Ge,
            FuzzInstruction::F64x2Eq => Instruction::F64x2Eq,
            FuzzInstruction::F64x2Ne => Instruction::F64x2Ne,
            FuzzInstruction::F64x2Lt => Instruction::F64x2Lt,
            FuzzInstruction::F64x2Gt => Instruction::F64x2Gt,
            FuzzInstruction::F64x2Le => Instruction::F64x2Le,
            FuzzInstruction::F64x2Ge => Instruction::F64x2Ge,
            FuzzInstruction::V128Not => Instruction::V128Not,
            FuzzInstruction::V128And => Instruction::V128And,
            FuzzInstruction::V128AndNot => Instruction::V128AndNot,
            FuzzInstruction::V128Or => Instruction::V128Or,
            FuzzInstruction::V128Xor => Instruction::V128Xor,
            FuzzInstruction::V128Bitselect => Instruction::V128Bitselect,
            FuzzInstruction::V128AnyTrue => Instruction::V128AnyTrue,
            FuzzInstruction::I8x16Abs => Instruction::I8x16Abs,
            FuzzInstruction::I8x16Neg => Instruction::I8x16Neg,
            FuzzInstruction::I8x16Popcnt => Instruction::I8x16Popcnt,
            FuzzInstruction::I8x16AllTrue => Instruction::I8x16AllTrue,
            FuzzInstruction::I8x16Bitmask => Instruction::I8x16Bitmask,
            FuzzInstruction::I8x16NarrowI16x8S => Instruction::I8x16NarrowI16x8S,
            FuzzInstruction::I8x16NarrowI16x8U => Instruction::I8x16NarrowI16x8U,
            FuzzInstruction::I8x16Shl => Instruction::I8x16Shl,
            FuzzInstruction::I8x16ShrS => Instruction::I8x16ShrS,
            FuzzInstruction::I8x16ShrU => Instruction::I8x16ShrU,
            FuzzInstruction::I8x16Add => Instruction::I8x16Add,
            FuzzInstruction::I8x16AddSatS => Instruction::I8x16AddSatS,
            FuzzInstruction::I8x16AddSatU => Instruction::I8x16AddSatU,
            FuzzInstruction::I8x16Sub => Instruction::I8x16Sub,
            FuzzInstruction::I8x16SubSatS => Instruction::I8x16SubSatS,
            FuzzInstruction::I8x16SubSatU => Instruction::I8x16SubSatU,
            FuzzInstruction::I8x16MinS => Instruction::I8x16MinS,
            FuzzInstruction::I8x16MinU => Instruction::I8x16MinU,
            FuzzInstruction::I8x16MaxS => Instruction::I8x16MaxS,
            FuzzInstruction::I8x16MaxU => Instruction::I8x16MaxU,
            FuzzInstruction::I8x16AvgrU => Instruction::I8x16AvgrU,
            FuzzInstruction::I16x8ExtAddPairwiseI8x16S => Instruction::I16x8ExtAddPairwiseI8x16S,
            FuzzInstruction::I16x8ExtAddPairwiseI8x16U => Instruction::I16x8ExtAddPairwiseI8x16U,
            FuzzInstruction::I16x8Abs => Instruction::I16x8Abs,
            FuzzInstruction::I16x8Neg => Instruction::I16x8Neg,
            FuzzInstruction::I16x8Q15MulrSatS => Instruction::I16x8Q15MulrSatS,
            FuzzInstruction::I16x8AllTrue => Instruction::I16x8AllTrue,
            FuzzInstruction::I16x8Bitmask => Instruction::I16x8Bitmask,
            FuzzInstruction::I16x8NarrowI32x4S => Instruction::I16x8NarrowI32x4S,
            FuzzInstruction::I16x8NarrowI32x4U => Instruction::I16x8NarrowI32x4U,
            FuzzInstruction::I16x8ExtendLowI8x16S => Instruction::I16x8ExtendLowI8x16S,
            FuzzInstruction::I16x8ExtendHighI8x16S => Instruction::I16x8ExtendHighI8x16S,
            FuzzInstruction::I16x8ExtendLowI8x16U => Instruction::I16x8ExtendLowI8x16U,
            FuzzInstruction::I16x8ExtendHighI8x16U => Instruction::I16x8ExtendHighI8x16U,
            FuzzInstruction::I16x8Shl => Instruction::I16x8Shl,
            FuzzInstruction::I16x8ShrS => Instruction::I16x8ShrS,
            FuzzInstruction::I16x8ShrU => Instruction::I16x8ShrU,
            FuzzInstruction::I16x8Add => Instruction::I16x8Add,
            FuzzInstruction::I16x8AddSatS => Instruction::I16x8AddSatS,
            FuzzInstruction::I16x8AddSatU => Instruction::I16x8AddSatU,
            FuzzInstruction::I16x8Sub => Instruction::I16x8Sub,
            FuzzInstruction::I16x8SubSatS => Instruction::I16x8SubSatS,
            FuzzInstruction::I16x8SubSatU => Instruction::I16x8SubSatU,
            FuzzInstruction::I16x8Mul => Instruction::I16x8Mul,
            FuzzInstruction::I16x8MinS => Instruction::I16x8MinS,
            FuzzInstruction::I16x8MinU => Instruction::I16x8MinU,
            FuzzInstruction::I16x8MaxS => Instruction::I16x8MaxS,
            FuzzInstruction::I16x8MaxU => Instruction::I16x8MaxU,
            FuzzInstruction::I16x8AvgrU => Instruction::I16x8AvgrU,
            FuzzInstruction::I16x8ExtMulLowI8x16S => Instruction::I16x8ExtMulLowI8x16S,
            FuzzInstruction::I16x8ExtMulHighI8x16S => Instruction::I16x8ExtMulHighI8x16S,
            FuzzInstruction::I16x8ExtMulLowI8x16U => Instruction::I16x8ExtMulLowI8x16U,
            FuzzInstruction::I16x8ExtMulHighI8x16U => Instruction::I16x8ExtMulHighI8x16U,
            FuzzInstruction::I32x4ExtAddPairwiseI16x8S => Instruction::I32x4ExtAddPairwiseI16x8S,
            FuzzInstruction::I32x4ExtAddPairwiseI16x8U => Instruction::I32x4ExtAddPairwiseI16x8U,
            FuzzInstruction::I32x4Abs => Instruction::I32x4Abs,
            FuzzInstruction::I32x4Neg => Instruction::I32x4Neg,
            FuzzInstruction::I32x4AllTrue => Instruction::I32x4AllTrue,
            FuzzInstruction::I32x4Bitmask => Instruction::I32x4Bitmask,
            FuzzInstruction::I32x4ExtendLowI16x8S => Instruction::I32x4ExtendLowI16x8S,
            FuzzInstruction::I32x4ExtendHighI16x8S => Instruction::I32x4ExtendHighI16x8S,
            FuzzInstruction::I32x4ExtendLowI16x8U => Instruction::I32x4ExtendLowI16x8U,
            FuzzInstruction::I32x4ExtendHighI16x8U => Instruction::I32x4ExtendHighI16x8U,
            FuzzInstruction::I32x4Shl => Instruction::I32x4Shl,
            FuzzInstruction::I32x4ShrS => Instruction::I32x4ShrS,
            FuzzInstruction::I32x4ShrU => Instruction::I32x4ShrU,
            FuzzInstruction::I32x4Add => Instruction::I32x4Add,
            FuzzInstruction::I32x4Sub => Instruction::I32x4Sub,
            FuzzInstruction::I32x4Mul => Instruction::I32x4Mul,
            FuzzInstruction::I32x4MinS => Instruction::I32x4MinS,
            FuzzInstruction::I32x4MinU => Instruction::I32x4MinU,
            FuzzInstruction::I32x4MaxS => Instruction::I32x4MaxS,
            FuzzInstruction::I32x4MaxU => Instruction::I32x4MaxU,
            FuzzInstruction::I32x4DotI16x8S => Instruction::I32x4DotI16x8S,
            FuzzInstruction::I32x4ExtMulLowI16x8S => Instruction::I32x4ExtMulLowI16x8S,
            FuzzInstruction::I32x4ExtMulHighI16x8S => Instruction::I32x4ExtMulHighI16x8S,
            FuzzInstruction::I32x4ExtMulLowI16x8U => Instruction::I32x4ExtMulLowI16x8U,
            FuzzInstruction::I32x4ExtMulHighI16x8U => Instruction::I32x4ExtMulHighI16x8U,
            FuzzInstruction::I64x2Abs => Instruction::I64x2Abs,
            FuzzInstruction::I64x2Neg => Instruction::I64x2Neg,
            FuzzInstruction::I64x2AllTrue => Instruction::I64x2AllTrue,
            FuzzInstruction::I64x2Bitmask => Instruction::I64x2Bitmask,
            FuzzInstruction::I64x2ExtendLowI32x4S => Instruction::I64x2ExtendLowI32x4S,
            FuzzInstruction::I64x2ExtendHighI32x4S => Instruction::I64x2ExtendHighI32x4S,
            FuzzInstruction::I64x2ExtendLowI32x4U => Instruction::I64x2ExtendLowI32x4U,
            FuzzInstruction::I64x2ExtendHighI32x4U => Instruction::I64x2ExtendHighI32x4U,
            FuzzInstruction::I64x2Shl => Instruction::I64x2Shl,
            FuzzInstruction::I64x2ShrS => Instruction::I64x2ShrS,
            FuzzInstruction::I64x2ShrU => Instruction::I64x2ShrU,
            FuzzInstruction::I64x2Add => Instruction::I64x2Add,
            FuzzInstruction::I64x2Sub => Instruction::I64x2Sub,
            FuzzInstruction::I64x2Mul => Instruction::I64x2Mul,
            FuzzInstruction::I64x2ExtMulLowI32x4S => Instruction::I64x2ExtMulLowI32x4S,
            FuzzInstruction::I64x2ExtMulHighI32x4S => Instruction::I64x2ExtMulHighI32x4S,
            FuzzInstruction::I64x2ExtMulLowI32x4U => Instruction::I64x2ExtMulLowI32x4U,
            FuzzInstruction::I64x2ExtMulHighI32x4U => Instruction::I64x2ExtMulHighI32x4U,
            FuzzInstruction::F32x4Ceil => Instruction::F32x4Ceil,
            FuzzInstruction::F32x4Floor => Instruction::F32x4Floor,
            FuzzInstruction::F32x4Trunc => Instruction::F32x4Trunc,
            FuzzInstruction::F32x4Nearest => Instruction::F32x4Nearest,
            FuzzInstruction::F32x4Abs => Instruction::F32x4Abs,
            FuzzInstruction::F32x4Neg => Instruction::F32x4Neg,
            FuzzInstruction::F32x4Sqrt => Instruction::F32x4Sqrt,
            FuzzInstruction::F32x4Add => Instruction::F32x4Add,
            FuzzInstruction::F32x4Sub => Instruction::F32x4Sub,
            FuzzInstruction::F32x4Mul => Instruction::F32x4Mul,
            FuzzInstruction::F32x4Div => Instruction::F32x4Div,
            FuzzInstruction::F32x4Min => Instruction::F32x4Min,
            FuzzInstruction::F32x4Max => Instruction::F32x4Max,
            FuzzInstruction::F32x4PMin => Instruction::F32x4PMin,
            FuzzInstruction::F32x4PMax => Instruction::F32x4PMax,
            FuzzInstruction::F64x2Ceil => Instruction::F64x2Ceil,
            FuzzInstruction::F64x2Floor => Instruction::F64x2Floor,
            FuzzInstruction::F64x2Trunc => Instruction::F64x2Trunc,
            FuzzInstruction::F64x2Nearest => Instruction::F64x2Nearest,
            FuzzInstruction::F64x2Abs => Instruction::F64x2Abs,
            FuzzInstruction::F64x2Neg => Instruction::F64x2Neg,
            FuzzInstruction::F64x2Sqrt => Instruction::F64x2Sqrt,
            FuzzInstruction::F64x2Add => Instruction::F64x2Add,
            FuzzInstruction::F64x2Sub => Instruction::F64x2Sub,
            FuzzInstruction::F64x2Mul => Instruction::F64x2Mul,
            FuzzInstruction::F64x2Div => Instruction::F64x2Div,
            FuzzInstruction::F64x2Min => Instruction::F64x2Min,
            FuzzInstruction::F64x2Max => Instruction::F64x2Max,
            FuzzInstruction::F64x2PMin => Instruction::F64x2PMin,
            FuzzInstruction::F64x2PMax => Instruction::F64x2PMax,
            FuzzInstruction::I32x4TruncSatF32x4S => Instruction::I32x4TruncSatF32x4S,
            FuzzInstruction::I32x4TruncSatF32x4U => Instruction::I32x4TruncSatF32x4U,
            FuzzInstruction::F32x4ConvertI32x4S => Instruction::F32x4ConvertI32x4S,
            FuzzInstruction::F32x4ConvertI32x4U => Instruction::F32x4ConvertI32x4U,
            FuzzInstruction::I32x4TruncSatF64x2SZero => Instruction::I32x4TruncSatF64x2SZero,
            FuzzInstruction::I32x4TruncSatF64x2UZero => Instruction::I32x4TruncSatF64x2UZero,
            FuzzInstruction::F64x2ConvertLowI32x4S => Instruction::F64x2ConvertLowI32x4S,
            FuzzInstruction::F64x2ConvertLowI32x4U => Instruction::F64x2ConvertLowI32x4U,
            FuzzInstruction::F32x4DemoteF64x2Zero => Instruction::F32x4DemoteF64x2Zero,
            FuzzInstruction::F64x2PromoteLowF32x4 => Instruction::F64x2PromoteLowF32x4,
            FuzzInstruction::I8x16RelaxedSwizzle => Instruction::I8x16RelaxedSwizzle,
            FuzzInstruction::I32x4RelaxedTruncF32x4S => Instruction::I32x4RelaxedTruncF32x4S,
            FuzzInstruction::I32x4RelaxedTruncF32x4U => Instruction::I32x4RelaxedTruncF32x4U,
            FuzzInstruction::I32x4RelaxedTruncF64x2SZero => {
                Instruction::I32x4RelaxedTruncF64x2SZero
            }
            FuzzInstruction::I32x4RelaxedTruncF64x2UZero => {
                Instruction::I32x4RelaxedTruncF64x2UZero
            }
            FuzzInstruction::F32x4RelaxedMadd => Instruction::F32x4RelaxedMadd,
            FuzzInstruction::F32x4RelaxedNmadd => Instruction::F32x4RelaxedNmadd,
            FuzzInstruction::F64x2RelaxedMadd => Instruction::F64x2RelaxedMadd,
            FuzzInstruction::F64x2RelaxedNmadd => Instruction::F64x2RelaxedNmadd,
            FuzzInstruction::I8x16RelaxedLaneselect => Instruction::I8x16RelaxedLaneselect,
            FuzzInstruction::I16x8RelaxedLaneselect => Instruction::I16x8RelaxedLaneselect,
            FuzzInstruction::I32x4RelaxedLaneselect => Instruction::I32x4RelaxedLaneselect,
            FuzzInstruction::I64x2RelaxedLaneselect => Instruction::I64x2RelaxedLaneselect,
            FuzzInstruction::F32x4RelaxedMin => Instruction::F32x4RelaxedMin,
            FuzzInstruction::F32x4RelaxedMax => Instruction::F32x4RelaxedMax,
            FuzzInstruction::F64x2RelaxedMin => Instruction::F64x2RelaxedMin,
            FuzzInstruction::F64x2RelaxedMax => Instruction::F64x2RelaxedMax,
            FuzzInstruction::I16x8RelaxedQ15mulrS => Instruction::I16x8RelaxedQ15mulrS,
            FuzzInstruction::I16x8RelaxedDotI8x16I7x16S => Instruction::I16x8RelaxedDotI8x16I7x16S,
            FuzzInstruction::I32x4RelaxedDotI8x16I7x16AddS => {
                Instruction::I32x4RelaxedDotI8x16I7x16AddS
            }
            FuzzInstruction::MemoryAtomicNotify(mem_arg) => {
                Instruction::MemoryAtomicNotify(mem_arg)
            }
            FuzzInstruction::MemoryAtomicWait32(mem_arg) => {
                Instruction::MemoryAtomicWait32(mem_arg)
            }
            FuzzInstruction::MemoryAtomicWait64(mem_arg) => {
                Instruction::MemoryAtomicWait64(mem_arg)
            }
            FuzzInstruction::AtomicFence => Instruction::AtomicFence,
            FuzzInstruction::I32AtomicLoad(mem_arg) => Instruction::I32AtomicLoad(mem_arg),
            FuzzInstruction::I64AtomicLoad(mem_arg) => Instruction::I64AtomicLoad(mem_arg),
            FuzzInstruction::I32AtomicLoad8U(mem_arg) => Instruction::I32AtomicLoad8U(mem_arg),
            FuzzInstruction::I32AtomicLoad16U(mem_arg) => Instruction::I32AtomicLoad16U(mem_arg),
            FuzzInstruction::I64AtomicLoad8U(mem_arg) => Instruction::I64AtomicLoad8U(mem_arg),
            FuzzInstruction::I64AtomicLoad16U(mem_arg) => Instruction::I64AtomicLoad16U(mem_arg),
            FuzzInstruction::I64AtomicLoad32U(mem_arg) => Instruction::I64AtomicLoad32U(mem_arg),
            FuzzInstruction::I32AtomicStore(mem_arg) => Instruction::I32AtomicStore(mem_arg),
            FuzzInstruction::I64AtomicStore(mem_arg) => Instruction::I64AtomicStore(mem_arg),
            FuzzInstruction::I32AtomicStore8(mem_arg) => Instruction::I32AtomicStore8(mem_arg),
            FuzzInstruction::I32AtomicStore16(mem_arg) => Instruction::I32AtomicStore16(mem_arg),
            FuzzInstruction::I64AtomicStore8(mem_arg) => Instruction::I64AtomicStore8(mem_arg),
            FuzzInstruction::I64AtomicStore16(mem_arg) => Instruction::I64AtomicStore16(mem_arg),
            FuzzInstruction::I64AtomicStore32(mem_arg) => Instruction::I64AtomicStore32(mem_arg),
            FuzzInstruction::I32AtomicRmwAdd(mem_arg) => Instruction::I32AtomicRmwAdd(mem_arg),
            FuzzInstruction::I64AtomicRmwAdd(mem_arg) => Instruction::I64AtomicRmwAdd(mem_arg),
            FuzzInstruction::I32AtomicRmw8AddU(mem_arg) => Instruction::I32AtomicRmw8AddU(mem_arg),
            FuzzInstruction::I32AtomicRmw16AddU(mem_arg) => {
                Instruction::I32AtomicRmw16AddU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw8AddU(mem_arg) => Instruction::I64AtomicRmw8AddU(mem_arg),
            FuzzInstruction::I64AtomicRmw16AddU(mem_arg) => {
                Instruction::I64AtomicRmw16AddU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw32AddU(mem_arg) => {
                Instruction::I64AtomicRmw32AddU(mem_arg)
            }
            FuzzInstruction::I32AtomicRmwSub(mem_arg) => Instruction::I32AtomicRmwSub(mem_arg),
            FuzzInstruction::I64AtomicRmwSub(mem_arg) => Instruction::I64AtomicRmwSub(mem_arg),
            FuzzInstruction::I32AtomicRmw8SubU(mem_arg) => Instruction::I32AtomicRmw8SubU(mem_arg),
            FuzzInstruction::I32AtomicRmw16SubU(mem_arg) => {
                Instruction::I32AtomicRmw16SubU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw8SubU(mem_arg) => Instruction::I64AtomicRmw8SubU(mem_arg),
            FuzzInstruction::I64AtomicRmw16SubU(mem_arg) => {
                Instruction::I64AtomicRmw16SubU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw32SubU(mem_arg) => {
                Instruction::I64AtomicRmw32SubU(mem_arg)
            }
            FuzzInstruction::I32AtomicRmwAnd(mem_arg) => Instruction::I32AtomicRmwAnd(mem_arg),
            FuzzInstruction::I64AtomicRmwAnd(mem_arg) => Instruction::I64AtomicRmwAnd(mem_arg),
            FuzzInstruction::I32AtomicRmw8AndU(mem_arg) => Instruction::I32AtomicRmw8AndU(mem_arg),
            FuzzInstruction::I32AtomicRmw16AndU(mem_arg) => {
                Instruction::I32AtomicRmw16AndU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw8AndU(mem_arg) => Instruction::I64AtomicRmw8AndU(mem_arg),
            FuzzInstruction::I64AtomicRmw16AndU(mem_arg) => {
                Instruction::I64AtomicRmw16AndU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw32AndU(mem_arg) => {
                Instruction::I64AtomicRmw32AndU(mem_arg)
            }
            FuzzInstruction::I32AtomicRmwOr(mem_arg) => Instruction::I32AtomicRmwOr(mem_arg),
            FuzzInstruction::I64AtomicRmwOr(mem_arg) => Instruction::I64AtomicRmwOr(mem_arg),
            FuzzInstruction::I32AtomicRmw8OrU(mem_arg) => Instruction::I32AtomicRmw8OrU(mem_arg),
            FuzzInstruction::I32AtomicRmw16OrU(mem_arg) => Instruction::I32AtomicRmw16OrU(mem_arg),
            FuzzInstruction::I64AtomicRmw8OrU(mem_arg) => Instruction::I64AtomicRmw8OrU(mem_arg),
            FuzzInstruction::I64AtomicRmw16OrU(mem_arg) => Instruction::I64AtomicRmw16OrU(mem_arg),
            FuzzInstruction::I64AtomicRmw32OrU(mem_arg) => Instruction::I64AtomicRmw32OrU(mem_arg),
            FuzzInstruction::I32AtomicRmwXor(mem_arg) => Instruction::I32AtomicRmwXor(mem_arg),
            FuzzInstruction::I64AtomicRmwXor(mem_arg) => Instruction::I64AtomicRmwXor(mem_arg),
            FuzzInstruction::I32AtomicRmw8XorU(mem_arg) => Instruction::I32AtomicRmw8XorU(mem_arg),
            FuzzInstruction::I32AtomicRmw16XorU(mem_arg) => {
                Instruction::I32AtomicRmw16XorU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw8XorU(mem_arg) => Instruction::I64AtomicRmw8XorU(mem_arg),
            FuzzInstruction::I64AtomicRmw16XorU(mem_arg) => {
                Instruction::I64AtomicRmw16XorU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw32XorU(mem_arg) => {
                Instruction::I64AtomicRmw32XorU(mem_arg)
            }
            FuzzInstruction::I32AtomicRmwXchg(mem_arg) => Instruction::I32AtomicRmwXchg(mem_arg),
            FuzzInstruction::I64AtomicRmwXchg(mem_arg) => Instruction::I64AtomicRmwXchg(mem_arg),
            FuzzInstruction::I32AtomicRmw8XchgU(mem_arg) => {
                Instruction::I32AtomicRmw8XchgU(mem_arg)
            }
            FuzzInstruction::I32AtomicRmw16XchgU(mem_arg) => {
                Instruction::I32AtomicRmw16XchgU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw8XchgU(mem_arg) => {
                Instruction::I64AtomicRmw8XchgU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw16XchgU(mem_arg) => {
                Instruction::I64AtomicRmw16XchgU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw32XchgU(mem_arg) => {
                Instruction::I64AtomicRmw32XchgU(mem_arg)
            }
            FuzzInstruction::I32AtomicRmwCmpxchg(mem_arg) => {
                Instruction::I32AtomicRmwCmpxchg(mem_arg)
            }
            FuzzInstruction::I64AtomicRmwCmpxchg(mem_arg) => {
                Instruction::I64AtomicRmwCmpxchg(mem_arg)
            }
            FuzzInstruction::I32AtomicRmw8CmpxchgU(mem_arg) => {
                Instruction::I32AtomicRmw8CmpxchgU(mem_arg)
            }
            FuzzInstruction::I32AtomicRmw16CmpxchgU(mem_arg) => {
                Instruction::I32AtomicRmw16CmpxchgU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw8CmpxchgU(mem_arg) => {
                Instruction::I64AtomicRmw8CmpxchgU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw16CmpxchgU(mem_arg) => {
                Instruction::I64AtomicRmw16CmpxchgU(mem_arg)
            }
            FuzzInstruction::I64AtomicRmw32CmpxchgU(mem_arg) => {
                Instruction::I64AtomicRmw32CmpxchgU(mem_arg)
            }

            // More atomic instructions
            FuzzInstruction::GlobalAtomicGet {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicGet {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicSet {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicSet {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicRmwAdd {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicRmwAdd {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicRmwSub {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicRmwSub {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicRmwAnd {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicRmwAnd {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicRmwOr {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicRmwOr {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicRmwXor {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicRmwXor {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicRmwXchg {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicRmwXchg {
                ordering,
                global_index,
            },
            FuzzInstruction::GlobalAtomicRmwCmpxchg {
                ordering,
                global_index,
            } => Instruction::GlobalAtomicRmwCmpxchg {
                ordering,
                global_index,
            },

            FuzzInstruction::TableAtomicGet {
                ordering,
                table_index,
            } => Instruction::TableAtomicGet {
                ordering,
                table_index,
            },
            FuzzInstruction::TableAtomicSet {
                ordering,
                table_index,
            } => Instruction::TableAtomicSet {
                ordering,
                table_index,
            },
            FuzzInstruction::TableAtomicRmwXchg {
                ordering,
                table_index,
            } => Instruction::TableAtomicRmwXchg {
                ordering,
                table_index,
            },
            FuzzInstruction::TableAtomicRmwCmpxchg {
                ordering,
                table_index,
            } => Instruction::TableAtomicRmwCmpxchg {
                ordering,
                table_index,
            },
            FuzzInstruction::StructAtomicGet {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicGet {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicGetS {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicGetS {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicGetU {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicGetU {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicSet {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicSet {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicRmwAdd {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicRmwAdd {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicRmwSub {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicRmwSub {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicRmwAnd {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicRmwAnd {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicRmwOr {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicRmwOr {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicRmwXor {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicRmwXor {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicRmwXchg {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicRmwXchg {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::StructAtomicRmwCmpxchg {
                ordering,
                struct_type_index,
                field_index,
            } => Instruction::StructAtomicRmwCmpxchg {
                ordering,
                struct_type_index,
                field_index,
            },
            FuzzInstruction::ArrayAtomicGet {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicGet {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicGetS {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicGetS {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicGetU {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicGetU {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicSet {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicSet {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicRmwAdd {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicRmwAdd {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicRmwSub {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicRmwSub {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicRmwAnd {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicRmwAnd {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicRmwOr {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicRmwOr {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicRmwXor {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicRmwXor {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicRmwXchg {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicRmwXchg {
                ordering,
                array_type_index,
            },
            FuzzInstruction::ArrayAtomicRmwCmpxchg {
                ordering,
                array_type_index,
            } => Instruction::ArrayAtomicRmwCmpxchg {
                ordering,
                array_type_index,
            },
            FuzzInstruction::RefI31Shared => Instruction::RefI31Shared,
            FuzzInstruction::ContNew(value) => Instruction::ContNew(value),
            FuzzInstruction::ContBind {
                argument_index,
                result_index,
            } => Instruction::ContBind {
                argument_index,
                result_index,
            },
            FuzzInstruction::Suspend(value) => Instruction::Suspend(value),
            FuzzInstruction::Resume {
                cont_type_index,
                resume_table,
            } => Instruction::Resume {
                cont_type_index,
                resume_table: resume_table.into(),
            },
            FuzzInstruction::ResumeThrow {
                cont_type_index,
                tag_index,
                resume_table,
            } => Instruction::ResumeThrow {
                cont_type_index,
                tag_index,
                resume_table: resume_table.into(),
            },
            FuzzInstruction::Switch {
                cont_type_index,
                tag_index,
            } => Instruction::Switch {
                cont_type_index,
                tag_index,
            },
            FuzzInstruction::I64Add128 => Instruction::I64Add128,
            FuzzInstruction::I64Sub128 => Instruction::I64Sub128,
            FuzzInstruction::I64MulWideS => Instruction::I64MulWideS,
            FuzzInstruction::I64MulWideU => Instruction::I64MulWideU,
        }
    }

}


/// WebAssembly instructions fuzzin'
#[derive(Debug, Clone, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum FuzzInstruction {
    // Control instructions.
    Unreachable,
    Nop,
    Block(BlockType),
    Loop(BlockType),
    If(BlockType),
    Else,
    End,
    Br(u32),
    BrIf(u32),
    BrTable(Vec<u32>, u32),
    BrOnNull(u32),
    BrOnNonNull(u32),
    Return,
    Call(u32),
    CallRef(u32),
    CallIndirect {
        type_index: u32,
        table_index: u32,
    },
    ReturnCallRef(u32),
    ReturnCall(u32),
    ReturnCallIndirect {
        type_index: u32,
        table_index: u32,
    },
    TryTable(BlockType, Vec<Catch>),
    Throw(u32),
    ThrowRef,

    // Deprecated exception-handling instructions
    Try(BlockType),
    Delegate(u32),
    Catch(u32),
    CatchAll,
    Rethrow(u32),

    // Parametric instructions.
    Drop,
    Select,

    // Variable instructions.
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    GlobalGet(u32),
    GlobalSet(u32),

    // Memory instructions.
    I32Load(MemArg),
    I64Load(MemArg),
    F32Load(MemArg),
    F64Load(MemArg),
    I32Load8S(MemArg),
    I32Load8U(MemArg),
    I32Load16S(MemArg),
    I32Load16U(MemArg),
    I64Load8S(MemArg),
    I64Load8U(MemArg),
    I64Load16S(MemArg),
    I64Load16U(MemArg),
    I64Load32S(MemArg),
    I64Load32U(MemArg),
    I32Store(MemArg),
    I64Store(MemArg),
    F32Store(MemArg),
    F64Store(MemArg),
    I32Store8(MemArg),
    I32Store16(MemArg),
    I64Store8(MemArg),
    I64Store16(MemArg),
    I64Store32(MemArg),
    MemorySize(u32),
    MemoryGrow(u32),
    MemoryInit {
        mem: u32,
        data_index: u32,
    },
    DataDrop(u32),
    MemoryCopy {
        src_mem: u32,
        dst_mem: u32,
    },
    MemoryFill(u32),
    MemoryDiscard(u32),

    // Numeric instructions.
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I64Eqz,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,
    F32Abs,
    F32Neg,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,
    F32Sqrt,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Copysign,
    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F64Sqrt,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,
    I32WrapI64,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64ExtendI32S,
    I64ExtendI32U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F32DemoteF64,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    F64PromoteF32,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,
    I32Extend8S,
    I32Extend16S,
    I64Extend8S,
    I64Extend16S,
    I64Extend32S,
    I32TruncSatF32S,
    I32TruncSatF32U,
    I32TruncSatF64S,
    I32TruncSatF64U,
    I64TruncSatF32S,
    I64TruncSatF32U,
    I64TruncSatF64S,
    I64TruncSatF64U,

    // Reference types instructions.
    TypedSelect(ValType),
    RefNull(HeapType),
    RefIsNull,
    RefFunc(u32),
    RefEq,
    RefAsNonNull,

    // GC types instructions.
    StructNew(u32),
    StructNewDefault(u32),
    StructGet {
        struct_type_index: u32,
        field_index: u32,
    },
    StructGetS {
        struct_type_index: u32,
        field_index: u32,
    },
    StructGetU {
        struct_type_index: u32,
        field_index: u32,
    },
    StructSet {
        struct_type_index: u32,
        field_index: u32,
    },

    ArrayNew(u32),
    ArrayNewDefault(u32),
    ArrayNewFixed {
        array_type_index: u32,
        array_size: u32,
    },
    ArrayNewData {
        array_type_index: u32,
        array_data_index: u32,
    },
    ArrayNewElem {
        array_type_index: u32,
        array_elem_index: u32,
    },
    ArrayGet(u32),
    ArrayGetS(u32),
    ArrayGetU(u32),
    ArraySet(u32),
    ArrayLen,
    ArrayFill(u32),
    ArrayCopy {
        array_type_index_dst: u32,
        array_type_index_src: u32,
    },
    ArrayInitData {
        array_type_index: u32,
        array_data_index: u32,
    },
    ArrayInitElem {
        array_type_index: u32,
        array_elem_index: u32,
    },
    RefTestNonNull(HeapType),
    RefTestNullable(HeapType),
    RefCastNonNull(HeapType),
    RefCastNullable(HeapType),
    BrOnCast {
        relative_depth: u32,
        from_ref_type: RefType,
        to_ref_type: RefType,
    },
    BrOnCastFail {
        relative_depth: u32,
        from_ref_type: RefType,
        to_ref_type: RefType,
    },
    AnyConvertExtern,
    ExternConvertAny,

    RefI31,
    I31GetS,
    I31GetU,

    // Bulk memory instructions.
    TableInit {
        elem_index: u32,
        table: u32,
    },
    ElemDrop(u32),
    TableFill(u32),
    TableSet(u32),
    TableGet(u32),
    TableGrow(u32),
    TableSize(u32),
    TableCopy {
        src_table: u32,
        dst_table: u32,
    },

    // SIMD instructions.
    V128Load(MemArg),
    V128Load8x8S(MemArg),
    V128Load8x8U(MemArg),
    V128Load16x4S(MemArg),
    V128Load16x4U(MemArg),
    V128Load32x2S(MemArg),
    V128Load32x2U(MemArg),
    V128Load8Splat(MemArg),
    V128Load16Splat(MemArg),
    V128Load32Splat(MemArg),
    V128Load64Splat(MemArg),
    V128Load32Zero(MemArg),
    V128Load64Zero(MemArg),
    V128Store(MemArg),
    V128Load8Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Load16Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Load32Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Load64Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store8Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store16Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store32Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store64Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Const(i128),
    I8x16Shuffle([Lane; 16]),
    I8x16ExtractLaneS(Lane),
    I8x16ExtractLaneU(Lane),
    I8x16ReplaceLane(Lane),
    I16x8ExtractLaneS(Lane),
    I16x8ExtractLaneU(Lane),
    I16x8ReplaceLane(Lane),
    I32x4ExtractLane(Lane),
    I32x4ReplaceLane(Lane),
    I64x2ExtractLane(Lane),
    I64x2ReplaceLane(Lane),
    F32x4ExtractLane(Lane),
    F32x4ReplaceLane(Lane),
    F64x2ExtractLane(Lane),
    F64x2ReplaceLane(Lane),
    I8x16Swizzle,
    I8x16Splat,
    I16x8Splat,
    I32x4Splat,
    I64x2Splat,
    F32x4Splat,
    F64x2Splat,
    I8x16Eq,
    I8x16Ne,
    I8x16LtS,
    I8x16LtU,
    I8x16GtS,
    I8x16GtU,
    I8x16LeS,
    I8x16LeU,
    I8x16GeS,
    I8x16GeU,
    I16x8Eq,
    I16x8Ne,
    I16x8LtS,
    I16x8LtU,
    I16x8GtS,
    I16x8GtU,
    I16x8LeS,
    I16x8LeU,
    I16x8GeS,
    I16x8GeU,
    I32x4Eq,
    I32x4Ne,
    I32x4LtS,
    I32x4LtU,
    I32x4GtS,
    I32x4GtU,
    I32x4LeS,
    I32x4LeU,
    I32x4GeS,
    I32x4GeU,
    I64x2Eq,
    I64x2Ne,
    I64x2LtS,
    I64x2GtS,
    I64x2LeS,
    I64x2GeS,
    F32x4Eq,
    F32x4Ne,
    F32x4Lt,
    F32x4Gt,
    F32x4Le,
    F32x4Ge,
    F64x2Eq,
    F64x2Ne,
    F64x2Lt,
    F64x2Gt,
    F64x2Le,
    F64x2Ge,
    V128Not,
    V128And,
    V128AndNot,
    V128Or,
    V128Xor,
    V128Bitselect,
    V128AnyTrue,
    I8x16Abs,
    I8x16Neg,
    I8x16Popcnt,
    I8x16AllTrue,
    I8x16Bitmask,
    I8x16NarrowI16x8S,
    I8x16NarrowI16x8U,
    I8x16Shl,
    I8x16ShrS,
    I8x16ShrU,
    I8x16Add,
    I8x16AddSatS,
    I8x16AddSatU,
    I8x16Sub,
    I8x16SubSatS,
    I8x16SubSatU,
    I8x16MinS,
    I8x16MinU,
    I8x16MaxS,
    I8x16MaxU,
    I8x16AvgrU,
    I16x8ExtAddPairwiseI8x16S,
    I16x8ExtAddPairwiseI8x16U,
    I16x8Abs,
    I16x8Neg,
    I16x8Q15MulrSatS,
    I16x8AllTrue,
    I16x8Bitmask,
    I16x8NarrowI32x4S,
    I16x8NarrowI32x4U,
    I16x8ExtendLowI8x16S,
    I16x8ExtendHighI8x16S,
    I16x8ExtendLowI8x16U,
    I16x8ExtendHighI8x16U,
    I16x8Shl,
    I16x8ShrS,
    I16x8ShrU,
    I16x8Add,
    I16x8AddSatS,
    I16x8AddSatU,
    I16x8Sub,
    I16x8SubSatS,
    I16x8SubSatU,
    I16x8Mul,
    I16x8MinS,
    I16x8MinU,
    I16x8MaxS,
    I16x8MaxU,
    I16x8AvgrU,
    I16x8ExtMulLowI8x16S,
    I16x8ExtMulHighI8x16S,
    I16x8ExtMulLowI8x16U,
    I16x8ExtMulHighI8x16U,
    I32x4ExtAddPairwiseI16x8S,
    I32x4ExtAddPairwiseI16x8U,
    I32x4Abs,
    I32x4Neg,
    I32x4AllTrue,
    I32x4Bitmask,
    I32x4ExtendLowI16x8S,
    I32x4ExtendHighI16x8S,
    I32x4ExtendLowI16x8U,
    I32x4ExtendHighI16x8U,
    I32x4Shl,
    I32x4ShrS,
    I32x4ShrU,
    I32x4Add,
    I32x4Sub,
    I32x4Mul,
    I32x4MinS,
    I32x4MinU,
    I32x4MaxS,
    I32x4MaxU,
    I32x4DotI16x8S,
    I32x4ExtMulLowI16x8S,
    I32x4ExtMulHighI16x8S,
    I32x4ExtMulLowI16x8U,
    I32x4ExtMulHighI16x8U,
    I64x2Abs,
    I64x2Neg,
    I64x2AllTrue,
    I64x2Bitmask,
    I64x2ExtendLowI32x4S,
    I64x2ExtendHighI32x4S,
    I64x2ExtendLowI32x4U,
    I64x2ExtendHighI32x4U,
    I64x2Shl,
    I64x2ShrS,
    I64x2ShrU,
    I64x2Add,
    I64x2Sub,
    I64x2Mul,
    I64x2ExtMulLowI32x4S,
    I64x2ExtMulHighI32x4S,
    I64x2ExtMulLowI32x4U,
    I64x2ExtMulHighI32x4U,
    F32x4Ceil,
    F32x4Floor,
    F32x4Trunc,
    F32x4Nearest,
    F32x4Abs,
    F32x4Neg,
    F32x4Sqrt,
    F32x4Add,
    F32x4Sub,
    F32x4Mul,
    F32x4Div,
    F32x4Min,
    F32x4Max,
    F32x4PMin,
    F32x4PMax,
    F64x2Ceil,
    F64x2Floor,
    F64x2Trunc,
    F64x2Nearest,
    F64x2Abs,
    F64x2Neg,
    F64x2Sqrt,
    F64x2Add,
    F64x2Sub,
    F64x2Mul,
    F64x2Div,
    F64x2Min,
    F64x2Max,
    F64x2PMin,
    F64x2PMax,
    I32x4TruncSatF32x4S,
    I32x4TruncSatF32x4U,
    F32x4ConvertI32x4S,
    F32x4ConvertI32x4U,
    I32x4TruncSatF64x2SZero,
    I32x4TruncSatF64x2UZero,
    F64x2ConvertLowI32x4S,
    F64x2ConvertLowI32x4U,
    F32x4DemoteF64x2Zero,
    F64x2PromoteLowF32x4,

    // Relaxed simd proposal
    I8x16RelaxedSwizzle,
    I32x4RelaxedTruncF32x4S,
    I32x4RelaxedTruncF32x4U,
    I32x4RelaxedTruncF64x2SZero,
    I32x4RelaxedTruncF64x2UZero,
    F32x4RelaxedMadd,
    F32x4RelaxedNmadd,
    F64x2RelaxedMadd,
    F64x2RelaxedNmadd,
    I8x16RelaxedLaneselect,
    I16x8RelaxedLaneselect,
    I32x4RelaxedLaneselect,
    I64x2RelaxedLaneselect,
    F32x4RelaxedMin,
    F32x4RelaxedMax,
    F64x2RelaxedMin,
    F64x2RelaxedMax,
    I16x8RelaxedQ15mulrS,
    I16x8RelaxedDotI8x16I7x16S,
    I32x4RelaxedDotI8x16I7x16AddS,

    // Atomic instructions (the threads proposal)
    MemoryAtomicNotify(MemArg),
    MemoryAtomicWait32(MemArg),
    MemoryAtomicWait64(MemArg),
    AtomicFence,
    I32AtomicLoad(MemArg),
    I64AtomicLoad(MemArg),
    I32AtomicLoad8U(MemArg),
    I32AtomicLoad16U(MemArg),
    I64AtomicLoad8U(MemArg),
    I64AtomicLoad16U(MemArg),
    I64AtomicLoad32U(MemArg),
    I32AtomicStore(MemArg),
    I64AtomicStore(MemArg),
    I32AtomicStore8(MemArg),
    I32AtomicStore16(MemArg),
    I64AtomicStore8(MemArg),
    I64AtomicStore16(MemArg),
    I64AtomicStore32(MemArg),
    I32AtomicRmwAdd(MemArg),
    I64AtomicRmwAdd(MemArg),
    I32AtomicRmw8AddU(MemArg),
    I32AtomicRmw16AddU(MemArg),
    I64AtomicRmw8AddU(MemArg),
    I64AtomicRmw16AddU(MemArg),
    I64AtomicRmw32AddU(MemArg),
    I32AtomicRmwSub(MemArg),
    I64AtomicRmwSub(MemArg),
    I32AtomicRmw8SubU(MemArg),
    I32AtomicRmw16SubU(MemArg),
    I64AtomicRmw8SubU(MemArg),
    I64AtomicRmw16SubU(MemArg),
    I64AtomicRmw32SubU(MemArg),
    I32AtomicRmwAnd(MemArg),
    I64AtomicRmwAnd(MemArg),
    I32AtomicRmw8AndU(MemArg),
    I32AtomicRmw16AndU(MemArg),
    I64AtomicRmw8AndU(MemArg),
    I64AtomicRmw16AndU(MemArg),
    I64AtomicRmw32AndU(MemArg),
    I32AtomicRmwOr(MemArg),
    I64AtomicRmwOr(MemArg),
    I32AtomicRmw8OrU(MemArg),
    I32AtomicRmw16OrU(MemArg),
    I64AtomicRmw8OrU(MemArg),
    I64AtomicRmw16OrU(MemArg),
    I64AtomicRmw32OrU(MemArg),
    I32AtomicRmwXor(MemArg),
    I64AtomicRmwXor(MemArg),
    I32AtomicRmw8XorU(MemArg),
    I32AtomicRmw16XorU(MemArg),
    I64AtomicRmw8XorU(MemArg),
    I64AtomicRmw16XorU(MemArg),
    I64AtomicRmw32XorU(MemArg),
    I32AtomicRmwXchg(MemArg),
    I64AtomicRmwXchg(MemArg),
    I32AtomicRmw8XchgU(MemArg),
    I32AtomicRmw16XchgU(MemArg),
    I64AtomicRmw8XchgU(MemArg),
    I64AtomicRmw16XchgU(MemArg),
    I64AtomicRmw32XchgU(MemArg),
    I32AtomicRmwCmpxchg(MemArg),
    I64AtomicRmwCmpxchg(MemArg),
    I32AtomicRmw8CmpxchgU(MemArg),
    I32AtomicRmw16CmpxchgU(MemArg),
    I64AtomicRmw8CmpxchgU(MemArg),
    I64AtomicRmw16CmpxchgU(MemArg),
    I64AtomicRmw32CmpxchgU(MemArg),

    // More atomic instructions (the shared-everything-threads proposal)
    GlobalAtomicGet {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicSet {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwAdd {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwSub {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwAnd {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwOr {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwXor {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwXchg {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwCmpxchg {
        ordering: Ordering,
        global_index: u32,
    },
    TableAtomicGet {
        ordering: Ordering,
        table_index: u32,
    },
    TableAtomicSet {
        ordering: Ordering,
        table_index: u32,
    },
    TableAtomicRmwXchg {
        ordering: Ordering,
        table_index: u32,
    },
    TableAtomicRmwCmpxchg {
        ordering: Ordering,
        table_index: u32,
    },
    StructAtomicGet {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicGetS {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicGetU {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicSet {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwAdd {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwSub {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwAnd {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwOr {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwXor {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwXchg {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwCmpxchg {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    ArrayAtomicGet {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicGetS {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicGetU {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicSet {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwAdd {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwSub {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwAnd {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwOr {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwXor {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwXchg {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwCmpxchg {
        ordering: Ordering,
        array_type_index: u32,
    },
    RefI31Shared,
    // Stack switching
    ContNew(u32),
    ContBind {
        argument_index: u32,
        result_index: u32,
    },
    Suspend(u32),
    Resume {
        cont_type_index: u32,
        resume_table: Vec<Handle>,
    },
    ResumeThrow {
        cont_type_index: u32,
        tag_index: u32,
        resume_table: Vec<Handle>,
    },
    Switch {
        cont_type_index: u32,
        tag_index: u32,
    },

    // Wide Arithmetic
    I64Add128,
    I64Sub128,
    I64MulWideS,
    I64MulWideU,
}

/// WebAssembly instructions.
#[derive(Clone, Debug, )]
#[non_exhaustive]
#[allow(missing_docs, non_camel_case_types)]
pub enum Instruction<'a> {
    // Control instructions.
    Unreachable,
    Nop,
    Block(BlockType),
    Loop(BlockType),
    If(BlockType),
    Else,
    End,
    Br(u32),
    BrIf(u32),
    BrTable(Cow<'a, [u32]>, u32),
    BrOnNull(u32),
    BrOnNonNull(u32),
    Return,
    Call(u32),
    CallRef(u32),
    CallIndirect {
        type_index: u32,
        table_index: u32,
    },
    ReturnCallRef(u32),
    ReturnCall(u32),
    ReturnCallIndirect {
        type_index: u32,
        table_index: u32,
    },
    TryTable(BlockType, Cow<'a, [Catch]>),
    Throw(u32),
    ThrowRef,

    // Deprecated exception-handling instructions
    Try(BlockType),
    Delegate(u32),
    Catch(u32),
    CatchAll,
    Rethrow(u32),

    // Parametric instructions.
    Drop,
    Select,

    // Variable instructions.
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    GlobalGet(u32),
    GlobalSet(u32),

    // Memory instructions.
    I32Load(MemArg),
    I64Load(MemArg),
    F32Load(MemArg),
    F64Load(MemArg),
    I32Load8S(MemArg),
    I32Load8U(MemArg),
    I32Load16S(MemArg),
    I32Load16U(MemArg),
    I64Load8S(MemArg),
    I64Load8U(MemArg),
    I64Load16S(MemArg),
    I64Load16U(MemArg),
    I64Load32S(MemArg),
    I64Load32U(MemArg),
    I32Store(MemArg),
    I64Store(MemArg),
    F32Store(MemArg),
    F64Store(MemArg),
    I32Store8(MemArg),
    I32Store16(MemArg),
    I64Store8(MemArg),
    I64Store16(MemArg),
    I64Store32(MemArg),
    MemorySize(u32),
    MemoryGrow(u32),
    MemoryInit {
        mem: u32,
        data_index: u32,
    },
    DataDrop(u32),
    MemoryCopy {
        src_mem: u32,
        dst_mem: u32,
    },
    MemoryFill(u32),
    MemoryDiscard(u32),

    // Numeric instructions.
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I64Eqz,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,
    F32Abs,
    F32Neg,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,
    F32Sqrt,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Copysign,
    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F64Sqrt,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,
    I32WrapI64,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64ExtendI32S,
    I64ExtendI32U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F32DemoteF64,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    F64PromoteF32,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,
    I32Extend8S,
    I32Extend16S,
    I64Extend8S,
    I64Extend16S,
    I64Extend32S,
    I32TruncSatF32S,
    I32TruncSatF32U,
    I32TruncSatF64S,
    I32TruncSatF64U,
    I64TruncSatF32S,
    I64TruncSatF32U,
    I64TruncSatF64S,
    I64TruncSatF64U,

    // Reference types instructions.
    TypedSelect(ValType),
    RefNull(HeapType),
    RefIsNull,
    RefFunc(u32),
    RefEq,
    RefAsNonNull,

    // GC types instructions.
    StructNew(u32),
    StructNewDefault(u32),
    StructGet {
        struct_type_index: u32,
        field_index: u32,
    },
    StructGetS {
        struct_type_index: u32,
        field_index: u32,
    },
    StructGetU {
        struct_type_index: u32,
        field_index: u32,
    },
    StructSet {
        struct_type_index: u32,
        field_index: u32,
    },

    ArrayNew(u32),
    ArrayNewDefault(u32),
    ArrayNewFixed {
        array_type_index: u32,
        array_size: u32,
    },
    ArrayNewData {
        array_type_index: u32,
        array_data_index: u32,
    },
    ArrayNewElem {
        array_type_index: u32,
        array_elem_index: u32,
    },
    ArrayGet(u32),
    ArrayGetS(u32),
    ArrayGetU(u32),
    ArraySet(u32),
    ArrayLen,
    ArrayFill(u32),
    ArrayCopy {
        array_type_index_dst: u32,
        array_type_index_src: u32,
    },
    ArrayInitData {
        array_type_index: u32,
        array_data_index: u32,
    },
    ArrayInitElem {
        array_type_index: u32,
        array_elem_index: u32,
    },
    RefTestNonNull(HeapType),
    RefTestNullable(HeapType),
    RefCastNonNull(HeapType),
    RefCastNullable(HeapType),
    BrOnCast {
        relative_depth: u32,
        from_ref_type: RefType,
        to_ref_type: RefType,
    },
    BrOnCastFail {
        relative_depth: u32,
        from_ref_type: RefType,
        to_ref_type: RefType,
    },
    AnyConvertExtern,
    ExternConvertAny,

    RefI31,
    I31GetS,
    I31GetU,

    // Bulk memory instructions.
    TableInit {
        elem_index: u32,
        table: u32,
    },
    ElemDrop(u32),
    TableFill(u32),
    TableSet(u32),
    TableGet(u32),
    TableGrow(u32),
    TableSize(u32),
    TableCopy {
        src_table: u32,
        dst_table: u32,
    },

    // SIMD instructions.
    V128Load(MemArg),
    V128Load8x8S(MemArg),
    V128Load8x8U(MemArg),
    V128Load16x4S(MemArg),
    V128Load16x4U(MemArg),
    V128Load32x2S(MemArg),
    V128Load32x2U(MemArg),
    V128Load8Splat(MemArg),
    V128Load16Splat(MemArg),
    V128Load32Splat(MemArg),
    V128Load64Splat(MemArg),
    V128Load32Zero(MemArg),
    V128Load64Zero(MemArg),
    V128Store(MemArg),
    V128Load8Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Load16Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Load32Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Load64Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store8Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store16Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store32Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Store64Lane {
        memarg: MemArg,
        lane: Lane,
    },
    V128Const(i128),
    I8x16Shuffle([Lane; 16]),
    I8x16ExtractLaneS(Lane),
    I8x16ExtractLaneU(Lane),
    I8x16ReplaceLane(Lane),
    I16x8ExtractLaneS(Lane),
    I16x8ExtractLaneU(Lane),
    I16x8ReplaceLane(Lane),
    I32x4ExtractLane(Lane),
    I32x4ReplaceLane(Lane),
    I64x2ExtractLane(Lane),
    I64x2ReplaceLane(Lane),
    F32x4ExtractLane(Lane),
    F32x4ReplaceLane(Lane),
    F64x2ExtractLane(Lane),
    F64x2ReplaceLane(Lane),
    I8x16Swizzle,
    I8x16Splat,
    I16x8Splat,
    I32x4Splat,
    I64x2Splat,
    F32x4Splat,
    F64x2Splat,
    I8x16Eq,
    I8x16Ne,
    I8x16LtS,
    I8x16LtU,
    I8x16GtS,
    I8x16GtU,
    I8x16LeS,
    I8x16LeU,
    I8x16GeS,
    I8x16GeU,
    I16x8Eq,
    I16x8Ne,
    I16x8LtS,
    I16x8LtU,
    I16x8GtS,
    I16x8GtU,
    I16x8LeS,
    I16x8LeU,
    I16x8GeS,
    I16x8GeU,
    I32x4Eq,
    I32x4Ne,
    I32x4LtS,
    I32x4LtU,
    I32x4GtS,
    I32x4GtU,
    I32x4LeS,
    I32x4LeU,
    I32x4GeS,
    I32x4GeU,
    I64x2Eq,
    I64x2Ne,
    I64x2LtS,
    I64x2GtS,
    I64x2LeS,
    I64x2GeS,
    F32x4Eq,
    F32x4Ne,
    F32x4Lt,
    F32x4Gt,
    F32x4Le,
    F32x4Ge,
    F64x2Eq,
    F64x2Ne,
    F64x2Lt,
    F64x2Gt,
    F64x2Le,
    F64x2Ge,
    V128Not,
    V128And,
    V128AndNot,
    V128Or,
    V128Xor,
    V128Bitselect,
    V128AnyTrue,
    I8x16Abs,
    I8x16Neg,
    I8x16Popcnt,
    I8x16AllTrue,
    I8x16Bitmask,
    I8x16NarrowI16x8S,
    I8x16NarrowI16x8U,
    I8x16Shl,
    I8x16ShrS,
    I8x16ShrU,
    I8x16Add,
    I8x16AddSatS,
    I8x16AddSatU,
    I8x16Sub,
    I8x16SubSatS,
    I8x16SubSatU,
    I8x16MinS,
    I8x16MinU,
    I8x16MaxS,
    I8x16MaxU,
    I8x16AvgrU,
    I16x8ExtAddPairwiseI8x16S,
    I16x8ExtAddPairwiseI8x16U,
    I16x8Abs,
    I16x8Neg,
    I16x8Q15MulrSatS,
    I16x8AllTrue,
    I16x8Bitmask,
    I16x8NarrowI32x4S,
    I16x8NarrowI32x4U,
    I16x8ExtendLowI8x16S,
    I16x8ExtendHighI8x16S,
    I16x8ExtendLowI8x16U,
    I16x8ExtendHighI8x16U,
    I16x8Shl,
    I16x8ShrS,
    I16x8ShrU,
    I16x8Add,
    I16x8AddSatS,
    I16x8AddSatU,
    I16x8Sub,
    I16x8SubSatS,
    I16x8SubSatU,
    I16x8Mul,
    I16x8MinS,
    I16x8MinU,
    I16x8MaxS,
    I16x8MaxU,
    I16x8AvgrU,
    I16x8ExtMulLowI8x16S,
    I16x8ExtMulHighI8x16S,
    I16x8ExtMulLowI8x16U,
    I16x8ExtMulHighI8x16U,
    I32x4ExtAddPairwiseI16x8S,
    I32x4ExtAddPairwiseI16x8U,
    I32x4Abs,
    I32x4Neg,
    I32x4AllTrue,
    I32x4Bitmask,
    I32x4ExtendLowI16x8S,
    I32x4ExtendHighI16x8S,
    I32x4ExtendLowI16x8U,
    I32x4ExtendHighI16x8U,
    I32x4Shl,
    I32x4ShrS,
    I32x4ShrU,
    I32x4Add,
    I32x4Sub,
    I32x4Mul,
    I32x4MinS,
    I32x4MinU,
    I32x4MaxS,
    I32x4MaxU,
    I32x4DotI16x8S,
    I32x4ExtMulLowI16x8S,
    I32x4ExtMulHighI16x8S,
    I32x4ExtMulLowI16x8U,
    I32x4ExtMulHighI16x8U,
    I64x2Abs,
    I64x2Neg,
    I64x2AllTrue,
    I64x2Bitmask,
    I64x2ExtendLowI32x4S,
    I64x2ExtendHighI32x4S,
    I64x2ExtendLowI32x4U,
    I64x2ExtendHighI32x4U,
    I64x2Shl,
    I64x2ShrS,
    I64x2ShrU,
    I64x2Add,
    I64x2Sub,
    I64x2Mul,
    I64x2ExtMulLowI32x4S,
    I64x2ExtMulHighI32x4S,
    I64x2ExtMulLowI32x4U,
    I64x2ExtMulHighI32x4U,
    F32x4Ceil,
    F32x4Floor,
    F32x4Trunc,
    F32x4Nearest,
    F32x4Abs,
    F32x4Neg,
    F32x4Sqrt,
    F32x4Add,
    F32x4Sub,
    F32x4Mul,
    F32x4Div,
    F32x4Min,
    F32x4Max,
    F32x4PMin,
    F32x4PMax,
    F64x2Ceil,
    F64x2Floor,
    F64x2Trunc,
    F64x2Nearest,
    F64x2Abs,
    F64x2Neg,
    F64x2Sqrt,
    F64x2Add,
    F64x2Sub,
    F64x2Mul,
    F64x2Div,
    F64x2Min,
    F64x2Max,
    F64x2PMin,
    F64x2PMax,
    I32x4TruncSatF32x4S,
    I32x4TruncSatF32x4U,
    F32x4ConvertI32x4S,
    F32x4ConvertI32x4U,
    I32x4TruncSatF64x2SZero,
    I32x4TruncSatF64x2UZero,
    F64x2ConvertLowI32x4S,
    F64x2ConvertLowI32x4U,
    F32x4DemoteF64x2Zero,
    F64x2PromoteLowF32x4,

    // Relaxed simd proposal
    I8x16RelaxedSwizzle,
    I32x4RelaxedTruncF32x4S,
    I32x4RelaxedTruncF32x4U,
    I32x4RelaxedTruncF64x2SZero,
    I32x4RelaxedTruncF64x2UZero,
    F32x4RelaxedMadd,
    F32x4RelaxedNmadd,
    F64x2RelaxedMadd,
    F64x2RelaxedNmadd,
    I8x16RelaxedLaneselect,
    I16x8RelaxedLaneselect,
    I32x4RelaxedLaneselect,
    I64x2RelaxedLaneselect,
    F32x4RelaxedMin,
    F32x4RelaxedMax,
    F64x2RelaxedMin,
    F64x2RelaxedMax,
    I16x8RelaxedQ15mulrS,
    I16x8RelaxedDotI8x16I7x16S,
    I32x4RelaxedDotI8x16I7x16AddS,

    // Atomic instructions (the threads proposal)
    MemoryAtomicNotify(MemArg),
    MemoryAtomicWait32(MemArg),
    MemoryAtomicWait64(MemArg),
    AtomicFence,
    I32AtomicLoad(MemArg),
    I64AtomicLoad(MemArg),
    I32AtomicLoad8U(MemArg),
    I32AtomicLoad16U(MemArg),
    I64AtomicLoad8U(MemArg),
    I64AtomicLoad16U(MemArg),
    I64AtomicLoad32U(MemArg),
    I32AtomicStore(MemArg),
    I64AtomicStore(MemArg),
    I32AtomicStore8(MemArg),
    I32AtomicStore16(MemArg),
    I64AtomicStore8(MemArg),
    I64AtomicStore16(MemArg),
    I64AtomicStore32(MemArg),
    I32AtomicRmwAdd(MemArg),
    I64AtomicRmwAdd(MemArg),
    I32AtomicRmw8AddU(MemArg),
    I32AtomicRmw16AddU(MemArg),
    I64AtomicRmw8AddU(MemArg),
    I64AtomicRmw16AddU(MemArg),
    I64AtomicRmw32AddU(MemArg),
    I32AtomicRmwSub(MemArg),
    I64AtomicRmwSub(MemArg),
    I32AtomicRmw8SubU(MemArg),
    I32AtomicRmw16SubU(MemArg),
    I64AtomicRmw8SubU(MemArg),
    I64AtomicRmw16SubU(MemArg),
    I64AtomicRmw32SubU(MemArg),
    I32AtomicRmwAnd(MemArg),
    I64AtomicRmwAnd(MemArg),
    I32AtomicRmw8AndU(MemArg),
    I32AtomicRmw16AndU(MemArg),
    I64AtomicRmw8AndU(MemArg),
    I64AtomicRmw16AndU(MemArg),
    I64AtomicRmw32AndU(MemArg),
    I32AtomicRmwOr(MemArg),
    I64AtomicRmwOr(MemArg),
    I32AtomicRmw8OrU(MemArg),
    I32AtomicRmw16OrU(MemArg),
    I64AtomicRmw8OrU(MemArg),
    I64AtomicRmw16OrU(MemArg),
    I64AtomicRmw32OrU(MemArg),
    I32AtomicRmwXor(MemArg),
    I64AtomicRmwXor(MemArg),
    I32AtomicRmw8XorU(MemArg),
    I32AtomicRmw16XorU(MemArg),
    I64AtomicRmw8XorU(MemArg),
    I64AtomicRmw16XorU(MemArg),
    I64AtomicRmw32XorU(MemArg),
    I32AtomicRmwXchg(MemArg),
    I64AtomicRmwXchg(MemArg),
    I32AtomicRmw8XchgU(MemArg),
    I32AtomicRmw16XchgU(MemArg),
    I64AtomicRmw8XchgU(MemArg),
    I64AtomicRmw16XchgU(MemArg),
    I64AtomicRmw32XchgU(MemArg),
    I32AtomicRmwCmpxchg(MemArg),
    I64AtomicRmwCmpxchg(MemArg),
    I32AtomicRmw8CmpxchgU(MemArg),
    I32AtomicRmw16CmpxchgU(MemArg),
    I64AtomicRmw8CmpxchgU(MemArg),
    I64AtomicRmw16CmpxchgU(MemArg),
    I64AtomicRmw32CmpxchgU(MemArg),

    // More atomic instructions (the shared-everything-threads proposal)
    GlobalAtomicGet {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicSet {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwAdd {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwSub {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwAnd {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwOr {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwXor {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwXchg {
        ordering: Ordering,
        global_index: u32,
    },
    GlobalAtomicRmwCmpxchg {
        ordering: Ordering,
        global_index: u32,
    },
    TableAtomicGet {
        ordering: Ordering,
        table_index: u32,
    },
    TableAtomicSet {
        ordering: Ordering,
        table_index: u32,
    },
    TableAtomicRmwXchg {
        ordering: Ordering,
        table_index: u32,
    },
    TableAtomicRmwCmpxchg {
        ordering: Ordering,
        table_index: u32,
    },
    StructAtomicGet {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicGetS {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicGetU {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicSet {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwAdd {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwSub {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwAnd {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwOr {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwXor {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwXchg {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    StructAtomicRmwCmpxchg {
        ordering: Ordering,
        struct_type_index: u32,
        field_index: u32,
    },
    ArrayAtomicGet {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicGetS {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicGetU {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicSet {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwAdd {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwSub {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwAnd {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwOr {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwXor {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwXchg {
        ordering: Ordering,
        array_type_index: u32,
    },
    ArrayAtomicRmwCmpxchg {
        ordering: Ordering,
        array_type_index: u32,
    },
    RefI31Shared,
    // Stack switching
    ContNew(u32),
    ContBind {
        argument_index: u32,
        result_index: u32,
    },
    Suspend(u32),
    Resume {
        cont_type_index: u32,
        resume_table: Cow<'a, [Handle]>,
    },
    ResumeThrow {
        cont_type_index: u32,
        tag_index: u32,
        resume_table: Cow<'a, [Handle]>,
    },
    Switch {
        cont_type_index: u32,
        tag_index: u32,
    },

    // Wide Arithmetic
    I64Add128,
    I64Sub128,
    I64MulWideS,
    I64MulWideU,
}

impl Encode for Instruction<'_> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        let mut sink = InstructionSink::new(bytes);
        match *self {
            // Control instructions.
            Instruction::Unreachable => sink.unreachable(),
            Instruction::Nop => sink.nop(),
            Instruction::Block(bt) => sink.block(bt),
            Instruction::Loop(bt) => sink.loop_(bt),
            Instruction::If(bt) => sink.if_(bt),
            Instruction::Else => sink.else_(),
            Instruction::Try(bt) => sink.try_(bt),
            Instruction::Catch(t) => sink.catch(t),
            Instruction::Throw(t) => sink.throw(t),
            Instruction::Rethrow(l) => sink.rethrow(l),
            Instruction::ThrowRef => sink.throw_ref(),
            Instruction::End => sink.end(),
            Instruction::Br(l) => sink.br(l),
            Instruction::BrIf(l) => sink.br_if(l),
            Instruction::BrTable(ref ls, l) => sink.br_table(ls.iter().copied(), l),
            Instruction::BrOnNull(l) => sink.br_on_null(l),
            Instruction::BrOnNonNull(l) => sink.br_on_non_null(l),
            Instruction::Return => sink.return_(),
            Instruction::Call(f) => sink.call(f),
            Instruction::CallRef(ty) => sink.call_ref(ty),
            Instruction::CallIndirect {
                type_index,
                table_index,
            } => sink.call_indirect(table_index, type_index),
            Instruction::ReturnCallRef(ty) => sink.return_call_ref(ty),

            Instruction::ReturnCall(f) => sink.return_call(f),
            Instruction::ReturnCallIndirect {
                type_index,
                table_index,
            } => sink.return_call_indirect(table_index, type_index),
            Instruction::Delegate(l) => sink.delegate(l),
            Instruction::CatchAll => sink.catch_all(),

            // Parametric instructions.
            Instruction::Drop => sink.drop(),
            Instruction::Select => sink.select(),
            Instruction::TypedSelect(ty) => sink.typed_select(ty),

            Instruction::TryTable(ty, ref catches) => sink.try_table(ty, catches.iter().cloned()),

            // Variable instructions.
            Instruction::LocalGet(l) => sink.local_get(l),
            Instruction::LocalSet(l) => sink.local_set(l),
            Instruction::LocalTee(l) => sink.local_tee(l),
            Instruction::GlobalGet(g) => sink.global_get(g),
            Instruction::GlobalSet(g) => sink.global_set(g),
            Instruction::TableGet(table) => sink.table_get(table),
            Instruction::TableSet(table) => sink.table_set(table),

            // Memory instructions.
            Instruction::I32Load(m) => sink.i32_load(m),
            Instruction::I64Load(m) => sink.i64_load(m),
            Instruction::F32Load(m) => sink.f32_load(m),
            Instruction::F64Load(m) => sink.f64_load(m),
            Instruction::I32Load8S(m) => sink.i32_load8_s(m),
            Instruction::I32Load8U(m) => sink.i32_load8_u(m),
            Instruction::I32Load16S(m) => sink.i32_load16_s(m),
            Instruction::I32Load16U(m) => sink.i32_load16_u(m),
            Instruction::I64Load8S(m) => sink.i64_load8_s(m),
            Instruction::I64Load8U(m) => sink.i64_load8_u(m),
            Instruction::I64Load16S(m) => sink.i64_load16_s(m),
            Instruction::I64Load16U(m) => sink.i64_load16_u(m),
            Instruction::I64Load32S(m) => sink.i64_load32_s(m),
            Instruction::I64Load32U(m) => sink.i64_load32_u(m),
            Instruction::I32Store(m) => sink.i32_store(m),
            Instruction::I64Store(m) => sink.i64_store(m),
            Instruction::F32Store(m) => sink.f32_store(m),
            Instruction::F64Store(m) => sink.f64_store(m),
            Instruction::I32Store8(m) => sink.i32_store8(m),
            Instruction::I32Store16(m) => sink.i32_store16(m),
            Instruction::I64Store8(m) => sink.i64_store8(m),
            Instruction::I64Store16(m) => sink.i64_store16(m),
            Instruction::I64Store32(m) => sink.i64_store32(m),
            Instruction::MemorySize(i) => sink.memory_size(i),
            Instruction::MemoryGrow(i) => sink.memory_grow(i),
            Instruction::MemoryInit { mem, data_index } => sink.memory_init(mem, data_index),
            Instruction::DataDrop(data) => sink.data_drop(data),
            Instruction::MemoryCopy { src_mem, dst_mem } => sink.memory_copy(dst_mem, src_mem),
            Instruction::MemoryFill(mem) => sink.memory_fill(mem),
            Instruction::MemoryDiscard(mem) => sink.memory_discard(mem),

            // Numeric instructions.
            Instruction::I32Const(x) => sink.i32_const(x),
            Instruction::I64Const(x) => sink.i64_const(x),
            Instruction::F32Const(x) => sink.f32_const(x),
            Instruction::F64Const(x) => sink.f64_const(x),
            Instruction::I32Eqz => sink.i32_eqz(),
            Instruction::I32Eq => sink.i32_eq(),
            Instruction::I32Ne => sink.i32_ne(),
            Instruction::I32LtS => sink.i32_lt_s(),
            Instruction::I32LtU => sink.i32_lt_u(),
            Instruction::I32GtS => sink.i32_gt_s(),
            Instruction::I32GtU => sink.i32_gt_u(),
            Instruction::I32LeS => sink.i32_le_s(),
            Instruction::I32LeU => sink.i32_le_u(),
            Instruction::I32GeS => sink.i32_ge_s(),
            Instruction::I32GeU => sink.i32_ge_u(),
            Instruction::I64Eqz => sink.i64_eqz(),
            Instruction::I64Eq => sink.i64_eq(),
            Instruction::I64Ne => sink.i64_ne(),
            Instruction::I64LtS => sink.i64_lt_s(),
            Instruction::I64LtU => sink.i64_lt_u(),
            Instruction::I64GtS => sink.i64_gt_s(),
            Instruction::I64GtU => sink.i64_gt_u(),
            Instruction::I64LeS => sink.i64_le_s(),
            Instruction::I64LeU => sink.i64_le_u(),
            Instruction::I64GeS => sink.i64_ge_s(),
            Instruction::I64GeU => sink.i64_ge_u(),
            Instruction::F32Eq => sink.f32_eq(),
            Instruction::F32Ne => sink.f32_ne(),
            Instruction::F32Lt => sink.f32_lt(),
            Instruction::F32Gt => sink.f32_gt(),
            Instruction::F32Le => sink.f32_le(),
            Instruction::F32Ge => sink.f32_ge(),
            Instruction::F64Eq => sink.f64_eq(),
            Instruction::F64Ne => sink.f64_ne(),
            Instruction::F64Lt => sink.f64_lt(),
            Instruction::F64Gt => sink.f64_gt(),
            Instruction::F64Le => sink.f64_le(),
            Instruction::F64Ge => sink.f64_ge(),
            Instruction::I32Clz => sink.i32_clz(),
            Instruction::I32Ctz => sink.i32_ctz(),
            Instruction::I32Popcnt => sink.i32_popcnt(),
            Instruction::I32Add => sink.i32_add(),
            Instruction::I32Sub => sink.i32_sub(),
            Instruction::I32Mul => sink.i32_mul(),
            Instruction::I32DivS => sink.i32_div_s(),
            Instruction::I32DivU => sink.i32_div_u(),
            Instruction::I32RemS => sink.i32_rem_s(),
            Instruction::I32RemU => sink.i32_rem_u(),
            Instruction::I32And => sink.i32_and(),
            Instruction::I32Or => sink.i32_or(),
            Instruction::I32Xor => sink.i32_xor(),
            Instruction::I32Shl => sink.i32_shl(),
            Instruction::I32ShrS => sink.i32_shr_s(),
            Instruction::I32ShrU => sink.i32_shr_u(),
            Instruction::I32Rotl => sink.i32_rotl(),
            Instruction::I32Rotr => sink.i32_rotr(),
            Instruction::I64Clz => sink.i64_clz(),
            Instruction::I64Ctz => sink.i64_ctz(),
            Instruction::I64Popcnt => sink.i64_popcnt(),
            Instruction::I64Add => sink.i64_add(),
            Instruction::I64Sub => sink.i64_sub(),
            Instruction::I64Mul => sink.i64_mul(),
            Instruction::I64DivS => sink.i64_div_s(),
            Instruction::I64DivU => sink.i64_div_u(),
            Instruction::I64RemS => sink.i64_rem_s(),
            Instruction::I64RemU => sink.i64_rem_u(),
            Instruction::I64And => sink.i64_and(),
            Instruction::I64Or => sink.i64_or(),
            Instruction::I64Xor => sink.i64_xor(),
            Instruction::I64Shl => sink.i64_shl(),
            Instruction::I64ShrS => sink.i64_shr_s(),
            Instruction::I64ShrU => sink.i64_shr_u(),
            Instruction::I64Rotl => sink.i64_rotl(),
            Instruction::I64Rotr => sink.i64_rotr(),
            Instruction::F32Abs => sink.f32_abs(),
            Instruction::F32Neg => sink.f32_neg(),
            Instruction::F32Ceil => sink.f32_ceil(),
            Instruction::F32Floor => sink.f32_floor(),
            Instruction::F32Trunc => sink.f32_trunc(),
            Instruction::F32Nearest => sink.f32_nearest(),
            Instruction::F32Sqrt => sink.f32_sqrt(),
            Instruction::F32Add => sink.f32_add(),
            Instruction::F32Sub => sink.f32_sub(),
            Instruction::F32Mul => sink.f32_mul(),
            Instruction::F32Div => sink.f32_div(),
            Instruction::F32Min => sink.f32_min(),
            Instruction::F32Max => sink.f32_max(),
            Instruction::F32Copysign => sink.f32_copysign(),
            Instruction::F64Abs => sink.f64_abs(),
            Instruction::F64Neg => sink.f64_neg(),
            Instruction::F64Ceil => sink.f64_ceil(),
            Instruction::F64Floor => sink.f64_floor(),
            Instruction::F64Trunc => sink.f64_trunc(),
            Instruction::F64Nearest => sink.f64_nearest(),
            Instruction::F64Sqrt => sink.f64_sqrt(),
            Instruction::F64Add => sink.f64_add(),
            Instruction::F64Sub => sink.f64_sub(),
            Instruction::F64Mul => sink.f64_mul(),
            Instruction::F64Div => sink.f64_div(),
            Instruction::F64Min => sink.f64_min(),
            Instruction::F64Max => sink.f64_max(),
            Instruction::F64Copysign => sink.f64_copysign(),
            Instruction::I32WrapI64 => sink.i32_wrap_i64(),
            Instruction::I32TruncF32S => sink.i32_trunc_f32_s(),
            Instruction::I32TruncF32U => sink.i32_trunc_f32_u(),
            Instruction::I32TruncF64S => sink.i32_trunc_f64_s(),
            Instruction::I32TruncF64U => sink.i32_trunc_f64_u(),
            Instruction::I64ExtendI32S => sink.i64_extend_i32_s(),
            Instruction::I64ExtendI32U => sink.i64_extend_i32_u(),
            Instruction::I64TruncF32S => sink.i64_trunc_f32_s(),
            Instruction::I64TruncF32U => sink.i64_trunc_f32_u(),
            Instruction::I64TruncF64S => sink.i64_trunc_f64_s(),
            Instruction::I64TruncF64U => sink.i64_trunc_f64_u(),
            Instruction::F32ConvertI32S => sink.f32_convert_i32_s(),
            Instruction::F32ConvertI32U => sink.f32_convert_i32_u(),
            Instruction::F32ConvertI64S => sink.f32_convert_i64_s(),
            Instruction::F32ConvertI64U => sink.f32_convert_i64_u(),
            Instruction::F32DemoteF64 => sink.f32_demote_f64(),
            Instruction::F64ConvertI32S => sink.f64_convert_i32_s(),
            Instruction::F64ConvertI32U => sink.f64_convert_i32_u(),
            Instruction::F64ConvertI64S => sink.f64_convert_i64_s(),
            Instruction::F64ConvertI64U => sink.f64_convert_i64_u(),
            Instruction::F64PromoteF32 => sink.f64_promote_f32(),
            Instruction::I32ReinterpretF32 => sink.i32_reinterpret_f32(),
            Instruction::I64ReinterpretF64 => sink.i64_reinterpret_f64(),
            Instruction::F32ReinterpretI32 => sink.f32_reinterpret_i32(),
            Instruction::F64ReinterpretI64 => sink.f64_reinterpret_i64(),
            Instruction::I32Extend8S => sink.i32_extend8_s(),
            Instruction::I32Extend16S => sink.i32_extend16_s(),
            Instruction::I64Extend8S => sink.i64_extend8_s(),
            Instruction::I64Extend16S => sink.i64_extend16_s(),
            Instruction::I64Extend32S => sink.i64_extend32_s(),

            Instruction::I32TruncSatF32S => sink.i32_trunc_sat_f32_s(),
            Instruction::I32TruncSatF32U => sink.i32_trunc_sat_f32_u(),
            Instruction::I32TruncSatF64S => sink.i32_trunc_sat_f64_s(),
            Instruction::I32TruncSatF64U => sink.i32_trunc_sat_f64_u(),
            Instruction::I64TruncSatF32S => sink.i64_trunc_sat_f32_s(),
            Instruction::I64TruncSatF32U => sink.i64_trunc_sat_f32_u(),
            Instruction::I64TruncSatF64S => sink.i64_trunc_sat_f64_s(),
            Instruction::I64TruncSatF64U => sink.i64_trunc_sat_f64_u(),

            // Reference types instructions.
            Instruction::RefNull(ty) => sink.ref_null(ty),
            Instruction::RefIsNull => sink.ref_is_null(),
            Instruction::RefFunc(f) => sink.ref_func(f),
            Instruction::RefEq => sink.ref_eq(),
            Instruction::RefAsNonNull => sink.ref_as_non_null(),

            // GC instructions.
            Instruction::StructNew(type_index) => sink.struct_new(type_index),
            Instruction::StructNewDefault(type_index) => sink.struct_new_default(type_index),
            Instruction::StructGet {
                struct_type_index,
                field_index,
            } => sink.struct_get(struct_type_index, field_index),
            Instruction::StructGetS {
                struct_type_index,
                field_index,
            } => sink.struct_get_s(struct_type_index, field_index),
            Instruction::StructGetU {
                struct_type_index,
                field_index,
            } => sink.struct_get_u(struct_type_index, field_index),
            Instruction::StructSet {
                struct_type_index,
                field_index,
            } => sink.struct_set(struct_type_index, field_index),
            Instruction::ArrayNew(type_index) => sink.array_new(type_index),
            Instruction::ArrayNewDefault(type_index) => sink.array_new_default(type_index),
            Instruction::ArrayNewFixed {
                array_type_index,
                array_size,
            } => sink.array_new_fixed(array_type_index, array_size),
            Instruction::ArrayNewData {
                array_type_index,
                array_data_index,
            } => sink.array_new_data(array_type_index, array_data_index),
            Instruction::ArrayNewElem {
                array_type_index,
                array_elem_index,
            } => sink.array_new_elem(array_type_index, array_elem_index),
            Instruction::ArrayGet(type_index) => sink.array_get(type_index),
            Instruction::ArrayGetS(type_index) => sink.array_get_s(type_index),
            Instruction::ArrayGetU(type_index) => sink.array_get_u(type_index),
            Instruction::ArraySet(type_index) => sink.array_set(type_index),
            Instruction::ArrayLen => sink.array_len(),
            Instruction::ArrayFill(type_index) => sink.array_fill(type_index),
            Instruction::ArrayCopy {
                array_type_index_dst,
                array_type_index_src,
            } => sink.array_copy(array_type_index_dst, array_type_index_src),
            Instruction::ArrayInitData {
                array_type_index,
                array_data_index,
            } => sink.array_init_data(array_type_index, array_data_index),
            Instruction::ArrayInitElem {
                array_type_index,
                array_elem_index,
            } => sink.array_init_elem(array_type_index, array_elem_index),
            Instruction::RefTestNonNull(heap_type) => sink.ref_test_non_null(heap_type),
            Instruction::RefTestNullable(heap_type) => sink.ref_test_nullable(heap_type),
            Instruction::RefCastNonNull(heap_type) => sink.ref_cast_non_null(heap_type),
            Instruction::RefCastNullable(heap_type) => sink.ref_cast_nullable(heap_type),
            Instruction::BrOnCast {
                relative_depth,
                from_ref_type,
                to_ref_type,
            } => sink.br_on_cast(relative_depth, from_ref_type, to_ref_type),
            Instruction::BrOnCastFail {
                relative_depth,
                from_ref_type,
                to_ref_type,
            } => sink.br_on_cast_fail(relative_depth, from_ref_type, to_ref_type),
            Instruction::AnyConvertExtern => sink.any_convert_extern(),
            Instruction::ExternConvertAny => sink.extern_convert_any(),
            Instruction::RefI31 => sink.ref_i31(),
            Instruction::I31GetS => sink.i31_get_s(),
            Instruction::I31GetU => sink.i31_get_u(),

            // Bulk memory instructions.
            Instruction::TableInit { elem_index, table } => sink.table_init(table, elem_index),
            Instruction::ElemDrop(segment) => sink.elem_drop(segment),
            Instruction::TableCopy {
                src_table,
                dst_table,
            } => sink.table_copy(dst_table, src_table),
            Instruction::TableGrow(table) => sink.table_grow(table),
            Instruction::TableSize(table) => sink.table_size(table),
            Instruction::TableFill(table) => sink.table_fill(table),

            // SIMD instructions.
            Instruction::V128Load(memarg) => sink.v128_load(memarg),
            Instruction::V128Load8x8S(memarg) => sink.v128_load8x8_s(memarg),
            Instruction::V128Load8x8U(memarg) => sink.v128_load8x8_u(memarg),
            Instruction::V128Load16x4S(memarg) => sink.v128_load16x4_s(memarg),
            Instruction::V128Load16x4U(memarg) => sink.v128_load16x4_u(memarg),
            Instruction::V128Load32x2S(memarg) => sink.v128_load32x2_s(memarg),
            Instruction::V128Load32x2U(memarg) => sink.v128_load32x2_u(memarg),
            Instruction::V128Load8Splat(memarg) => sink.v128_load8_splat(memarg),
            Instruction::V128Load16Splat(memarg) => sink.v128_load16_splat(memarg),
            Instruction::V128Load32Splat(memarg) => sink.v128_load32_splat(memarg),
            Instruction::V128Load64Splat(memarg) => sink.v128_load64_splat(memarg),
            Instruction::V128Store(memarg) => sink.v128_store(memarg),
            Instruction::V128Const(x) => sink.v128_const(x),
            Instruction::I8x16Shuffle(lanes) => sink.i8x16_shuffle(lanes),
            Instruction::I8x16Swizzle => sink.i8x16_swizzle(),
            Instruction::I8x16Splat => sink.i8x16_splat(),
            Instruction::I16x8Splat => sink.i16x8_splat(),
            Instruction::I32x4Splat => sink.i32x4_splat(),
            Instruction::I64x2Splat => sink.i64x2_splat(),
            Instruction::F32x4Splat => sink.f32x4_splat(),
            Instruction::F64x2Splat => sink.f64x2_splat(),
            Instruction::I8x16ExtractLaneS(lane) => sink.i8x16_extract_lane_s(lane),
            Instruction::I8x16ExtractLaneU(lane) => sink.i8x16_extract_lane_u(lane),
            Instruction::I8x16ReplaceLane(lane) => sink.i8x16_replace_lane(lane),
            Instruction::I16x8ExtractLaneS(lane) => sink.i16x8_extract_lane_s(lane),
            Instruction::I16x8ExtractLaneU(lane) => sink.i16x8_extract_lane_u(lane),
            Instruction::I16x8ReplaceLane(lane) => sink.i16x8_replace_lane(lane),
            Instruction::I32x4ExtractLane(lane) => sink.i32x4_extract_lane(lane),
            Instruction::I32x4ReplaceLane(lane) => sink.i32x4_replace_lane(lane),
            Instruction::I64x2ExtractLane(lane) => sink.i64x2_extract_lane(lane),
            Instruction::I64x2ReplaceLane(lane) => sink.i64x2_replace_lane(lane),
            Instruction::F32x4ExtractLane(lane) => sink.f32x4_extract_lane(lane),
            Instruction::F32x4ReplaceLane(lane) => sink.f32x4_replace_lane(lane),
            Instruction::F64x2ExtractLane(lane) => sink.f64x2_extract_lane(lane),
            Instruction::F64x2ReplaceLane(lane) => sink.f64x2_replace_lane(lane),

            Instruction::I8x16Eq => sink.i8x16_eq(),
            Instruction::I8x16Ne => sink.i8x16_ne(),
            Instruction::I8x16LtS => sink.i8x16_lt_s(),
            Instruction::I8x16LtU => sink.i8x16_lt_u(),
            Instruction::I8x16GtS => sink.i8x16_gt_s(),
            Instruction::I8x16GtU => sink.i8x16_gt_u(),
            Instruction::I8x16LeS => sink.i8x16_le_s(),
            Instruction::I8x16LeU => sink.i8x16_le_u(),
            Instruction::I8x16GeS => sink.i8x16_ge_s(),
            Instruction::I8x16GeU => sink.i8x16_ge_u(),
            Instruction::I16x8Eq => sink.i16x8_eq(),
            Instruction::I16x8Ne => sink.i16x8_ne(),
            Instruction::I16x8LtS => sink.i16x8_lt_s(),
            Instruction::I16x8LtU => sink.i16x8_lt_u(),
            Instruction::I16x8GtS => sink.i16x8_gt_s(),
            Instruction::I16x8GtU => sink.i16x8_gt_u(),
            Instruction::I16x8LeS => sink.i16x8_le_s(),
            Instruction::I16x8LeU => sink.i16x8_le_u(),
            Instruction::I16x8GeS => sink.i16x8_ge_s(),
            Instruction::I16x8GeU => sink.i16x8_ge_u(),
            Instruction::I32x4Eq => sink.i32x4_eq(),
            Instruction::I32x4Ne => sink.i32x4_ne(),
            Instruction::I32x4LtS => sink.i32x4_lt_s(),
            Instruction::I32x4LtU => sink.i32x4_lt_u(),
            Instruction::I32x4GtS => sink.i32x4_gt_s(),
            Instruction::I32x4GtU => sink.i32x4_gt_u(),
            Instruction::I32x4LeS => sink.i32x4_le_s(),
            Instruction::I32x4LeU => sink.i32x4_le_u(),
            Instruction::I32x4GeS => sink.i32x4_ge_s(),
            Instruction::I32x4GeU => sink.i32x4_ge_u(),
            Instruction::F32x4Eq => sink.f32x4_eq(),
            Instruction::F32x4Ne => sink.f32x4_ne(),
            Instruction::F32x4Lt => sink.f32x4_lt(),
            Instruction::F32x4Gt => sink.f32x4_gt(),
            Instruction::F32x4Le => sink.f32x4_le(),
            Instruction::F32x4Ge => sink.f32x4_ge(),
            Instruction::F64x2Eq => sink.f64x2_eq(),
            Instruction::F64x2Ne => sink.f64x2_ne(),
            Instruction::F64x2Lt => sink.f64x2_lt(),
            Instruction::F64x2Gt => sink.f64x2_gt(),
            Instruction::F64x2Le => sink.f64x2_le(),
            Instruction::F64x2Ge => sink.f64x2_ge(),
            Instruction::V128Not => sink.v128_not(),
            Instruction::V128And => sink.v128_and(),
            Instruction::V128AndNot => sink.v128_andnot(),
            Instruction::V128Or => sink.v128_or(),
            Instruction::V128Xor => sink.v128_xor(),
            Instruction::V128Bitselect => sink.v128_bitselect(),
            Instruction::V128AnyTrue => sink.v128_any_true(),
            Instruction::I8x16Abs => sink.i8x16_abs(),
            Instruction::I8x16Neg => sink.i8x16_neg(),
            Instruction::I8x16Popcnt => sink.i8x16_popcnt(),
            Instruction::I8x16AllTrue => sink.i8x16_all_true(),
            Instruction::I8x16Bitmask => sink.i8x16_bitmask(),
            Instruction::I8x16NarrowI16x8S => sink.i8x16_narrow_i16x8_s(),
            Instruction::I8x16NarrowI16x8U => sink.i8x16_narrow_i16x8_u(),
            Instruction::I8x16Shl => sink.i8x16_shl(),
            Instruction::I8x16ShrS => sink.i8x16_shr_s(),
            Instruction::I8x16ShrU => sink.i8x16_shr_u(),
            Instruction::I8x16Add => sink.i8x16_add(),
            Instruction::I8x16AddSatS => sink.i8x16_add_sat_s(),
            Instruction::I8x16AddSatU => sink.i8x16_add_sat_u(),
            Instruction::I8x16Sub => sink.i8x16_sub(),
            Instruction::I8x16SubSatS => sink.i8x16_sub_sat_s(),
            Instruction::I8x16SubSatU => sink.i8x16_sub_sat_u(),
            Instruction::I8x16MinS => sink.i8x16_min_s(),
            Instruction::I8x16MinU => sink.i8x16_min_u(),
            Instruction::I8x16MaxS => sink.i8x16_max_s(),
            Instruction::I8x16MaxU => sink.i8x16_max_u(),
            Instruction::I8x16AvgrU => sink.i8x16_avgr_u(),
            Instruction::I16x8ExtAddPairwiseI8x16S => sink.i16x8_extadd_pairwise_i8x16_s(),
            Instruction::I16x8ExtAddPairwiseI8x16U => sink.i16x8_extadd_pairwise_i8x16_u(),
            Instruction::I32x4ExtAddPairwiseI16x8S => sink.i32x4_extadd_pairwise_i16x8_s(),
            Instruction::I32x4ExtAddPairwiseI16x8U => sink.i32x4_extadd_pairwise_i16x8_u(),
            Instruction::I16x8Abs => sink.i16x8_abs(),
            Instruction::I16x8Neg => sink.i16x8_neg(),
            Instruction::I16x8Q15MulrSatS => sink.i16x8_q15mulr_sat_s(),
            Instruction::I16x8AllTrue => sink.i16x8_all_true(),
            Instruction::I16x8Bitmask => sink.i16x8_bitmask(),
            Instruction::I16x8NarrowI32x4S => sink.i16x8_narrow_i32x4_s(),
            Instruction::I16x8NarrowI32x4U => sink.i16x8_narrow_i32x4_u(),
            Instruction::I16x8ExtendLowI8x16S => sink.i16x8_extend_low_i8x16_s(),
            Instruction::I16x8ExtendHighI8x16S => sink.i16x8_extend_high_i8x16_s(),
            Instruction::I16x8ExtendLowI8x16U => sink.i16x8_extend_low_i8x16_u(),
            Instruction::I16x8ExtendHighI8x16U => sink.i16x8_extend_high_i8x16_u(),
            Instruction::I16x8Shl => sink.i16x8_shl(),
            Instruction::I16x8ShrS => sink.i16x8_shr_s(),
            Instruction::I16x8ShrU => sink.i16x8_shr_u(),
            Instruction::I16x8Add => sink.i16x8_add(),
            Instruction::I16x8AddSatS => sink.i16x8_add_sat_s(),
            Instruction::I16x8AddSatU => sink.i16x8_add_sat_u(),
            Instruction::I16x8Sub => sink.i16x8_sub(),
            Instruction::I16x8SubSatS => sink.i16x8_sub_sat_s(),
            Instruction::I16x8SubSatU => sink.i16x8_sub_sat_u(),
            Instruction::I16x8Mul => sink.i16x8_mul(),
            Instruction::I16x8MinS => sink.i16x8_min_s(),
            Instruction::I16x8MinU => sink.i16x8_min_u(),
            Instruction::I16x8MaxS => sink.i16x8_max_s(),
            Instruction::I16x8MaxU => sink.i16x8_max_u(),
            Instruction::I16x8AvgrU => sink.i16x8_avgr_u(),
            Instruction::I16x8ExtMulLowI8x16S => sink.i16x8_extmul_low_i8x16_s(),
            Instruction::I16x8ExtMulHighI8x16S => sink.i16x8_extmul_high_i8x16_s(),
            Instruction::I16x8ExtMulLowI8x16U => sink.i16x8_extmul_low_i8x16_u(),
            Instruction::I16x8ExtMulHighI8x16U => sink.i16x8_extmul_high_i8x16_u(),
            Instruction::I32x4Abs => sink.i32x4_abs(),
            Instruction::I32x4Neg => sink.i32x4_neg(),
            Instruction::I32x4AllTrue => sink.i32x4_all_true(),
            Instruction::I32x4Bitmask => sink.i32x4_bitmask(),
            Instruction::I32x4ExtendLowI16x8S => sink.i32x4_extend_low_i16x8_s(),
            Instruction::I32x4ExtendHighI16x8S => sink.i32x4_extend_high_i16x8_s(),
            Instruction::I32x4ExtendLowI16x8U => sink.i32x4_extend_low_i16x8_u(),
            Instruction::I32x4ExtendHighI16x8U => sink.i32x4_extend_high_i16x8_u(),
            Instruction::I32x4Shl => sink.i32x4_shl(),
            Instruction::I32x4ShrS => sink.i32x4_shr_s(),
            Instruction::I32x4ShrU => sink.i32x4_shr_u(),
            Instruction::I32x4Add => sink.i32x4_add(),
            Instruction::I32x4Sub => sink.i32x4_sub(),
            Instruction::I32x4Mul => sink.i32x4_mul(),
            Instruction::I32x4MinS => sink.i32x4_min_s(),
            Instruction::I32x4MinU => sink.i32x4_min_u(),
            Instruction::I32x4MaxS => sink.i32x4_max_s(),
            Instruction::I32x4MaxU => sink.i32x4_max_u(),
            Instruction::I32x4DotI16x8S => sink.i32x4_dot_i16x8_s(),
            Instruction::I32x4ExtMulLowI16x8S => sink.i32x4_extmul_low_i16x8_s(),
            Instruction::I32x4ExtMulHighI16x8S => sink.i32x4_extmul_high_i16x8_s(),
            Instruction::I32x4ExtMulLowI16x8U => sink.i32x4_extmul_low_i16x8_u(),
            Instruction::I32x4ExtMulHighI16x8U => sink.i32x4_extmul_high_i16x8_u(),
            Instruction::I64x2Abs => sink.i64x2_abs(),
            Instruction::I64x2Neg => sink.i64x2_neg(),
            Instruction::I64x2AllTrue => sink.i64x2_all_true(),
            Instruction::I64x2Bitmask => sink.i64x2_bitmask(),
            Instruction::I64x2ExtendLowI32x4S => sink.i64x2_extend_low_i32x4_s(),
            Instruction::I64x2ExtendHighI32x4S => sink.i64x2_extend_high_i32x4_s(),
            Instruction::I64x2ExtendLowI32x4U => sink.i64x2_extend_low_i32x4_u(),
            Instruction::I64x2ExtendHighI32x4U => sink.i64x2_extend_high_i32x4_u(),
            Instruction::I64x2Shl => sink.i64x2_shl(),
            Instruction::I64x2ShrS => sink.i64x2_shr_s(),
            Instruction::I64x2ShrU => sink.i64x2_shr_u(),
            Instruction::I64x2Add => sink.i64x2_add(),
            Instruction::I64x2Sub => sink.i64x2_sub(),
            Instruction::I64x2Mul => sink.i64x2_mul(),
            Instruction::I64x2ExtMulLowI32x4S => sink.i64x2_extmul_low_i32x4_s(),
            Instruction::I64x2ExtMulHighI32x4S => sink.i64x2_extmul_high_i32x4_s(),
            Instruction::I64x2ExtMulLowI32x4U => sink.i64x2_extmul_low_i32x4_u(),
            Instruction::I64x2ExtMulHighI32x4U => sink.i64x2_extmul_high_i32x4_u(),
            Instruction::F32x4Ceil => sink.f32x4_ceil(),
            Instruction::F32x4Floor => sink.f32x4_floor(),
            Instruction::F32x4Trunc => sink.f32x4_trunc(),
            Instruction::F32x4Nearest => sink.f32x4_nearest(),
            Instruction::F32x4Abs => sink.f32x4_abs(),
            Instruction::F32x4Neg => sink.f32x4_neg(),
            Instruction::F32x4Sqrt => sink.f32x4_sqrt(),
            Instruction::F32x4Add => sink.f32x4_add(),
            Instruction::F32x4Sub => sink.f32x4_sub(),
            Instruction::F32x4Mul => sink.f32x4_mul(),
            Instruction::F32x4Div => sink.f32x4_div(),
            Instruction::F32x4Min => sink.f32x4_min(),
            Instruction::F32x4Max => sink.f32x4_max(),
            Instruction::F32x4PMin => sink.f32x4_pmin(),
            Instruction::F32x4PMax => sink.f32x4_pmax(),
            Instruction::F64x2Ceil => sink.f64x2_ceil(),
            Instruction::F64x2Floor => sink.f64x2_floor(),
            Instruction::F64x2Trunc => sink.f64x2_trunc(),
            Instruction::F64x2Nearest => sink.f64x2_nearest(),
            Instruction::F64x2Abs => sink.f64x2_abs(),
            Instruction::F64x2Neg => sink.f64x2_neg(),
            Instruction::F64x2Sqrt => sink.f64x2_sqrt(),
            Instruction::F64x2Add => sink.f64x2_add(),
            Instruction::F64x2Sub => sink.f64x2_sub(),
            Instruction::F64x2Mul => sink.f64x2_mul(),
            Instruction::F64x2Div => sink.f64x2_div(),
            Instruction::F64x2Min => sink.f64x2_min(),
            Instruction::F64x2Max => sink.f64x2_max(),
            Instruction::F64x2PMin => sink.f64x2_pmin(),
            Instruction::F64x2PMax => sink.f64x2_pmax(),
            Instruction::I32x4TruncSatF32x4S => sink.i32x4_trunc_sat_f32x4_s(),
            Instruction::I32x4TruncSatF32x4U => sink.i32x4_trunc_sat_f32x4_u(),
            Instruction::F32x4ConvertI32x4S => sink.f32x4_convert_i32x4_s(),
            Instruction::F32x4ConvertI32x4U => sink.f32x4_convert_i32x4_u(),
            Instruction::I32x4TruncSatF64x2SZero => sink.i32x4_trunc_sat_f64x2_s_zero(),
            Instruction::I32x4TruncSatF64x2UZero => sink.i32x4_trunc_sat_f64x2_u_zero(),
            Instruction::F64x2ConvertLowI32x4S => sink.f64x2_convert_low_i32x4_s(),
            Instruction::F64x2ConvertLowI32x4U => sink.f64x2_convert_low_i32x4_u(),
            Instruction::F32x4DemoteF64x2Zero => sink.f32x4_demote_f64x2_zero(),
            Instruction::F64x2PromoteLowF32x4 => sink.f64x2_promote_low_f32x4(),
            Instruction::V128Load32Zero(memarg) => sink.v128_load32_zero(memarg),
            Instruction::V128Load64Zero(memarg) => sink.v128_load64_zero(memarg),
            Instruction::V128Load8Lane { memarg, lane } => sink.v128_load8_lane(memarg, lane),
            Instruction::V128Load16Lane { memarg, lane } => sink.v128_load16_lane(memarg, lane),
            Instruction::V128Load32Lane { memarg, lane } => sink.v128_load32_lane(memarg, lane),
            Instruction::V128Load64Lane { memarg, lane } => sink.v128_load64_lane(memarg, lane),
            Instruction::V128Store8Lane { memarg, lane } => sink.v128_store8_lane(memarg, lane),
            Instruction::V128Store16Lane { memarg, lane } => sink.v128_store16_lane(memarg, lane),
            Instruction::V128Store32Lane { memarg, lane } => sink.v128_store32_lane(memarg, lane),
            Instruction::V128Store64Lane { memarg, lane } => sink.v128_store64_lane(memarg, lane),
            Instruction::I64x2Eq => sink.i64x2_eq(),
            Instruction::I64x2Ne => sink.i64x2_ne(),
            Instruction::I64x2LtS => sink.i64x2_lt_s(),
            Instruction::I64x2GtS => sink.i64x2_gt_s(),
            Instruction::I64x2LeS => sink.i64x2_le_s(),
            Instruction::I64x2GeS => sink.i64x2_ge_s(),
            Instruction::I8x16RelaxedSwizzle => sink.i8x16_relaxed_swizzle(),
            Instruction::I32x4RelaxedTruncF32x4S => sink.i32x4_relaxed_trunc_f32x4_s(),
            Instruction::I32x4RelaxedTruncF32x4U => sink.i32x4_relaxed_trunc_f32x4_u(),
            Instruction::I32x4RelaxedTruncF64x2SZero => sink.i32x4_relaxed_trunc_f64x2_s_zero(),
            Instruction::I32x4RelaxedTruncF64x2UZero => sink.i32x4_relaxed_trunc_f64x2_u_zero(),
            Instruction::F32x4RelaxedMadd => sink.f32x4_relaxed_madd(),
            Instruction::F32x4RelaxedNmadd => sink.f32x4_relaxed_nmadd(),
            Instruction::F64x2RelaxedMadd => sink.f64x2_relaxed_madd(),
            Instruction::F64x2RelaxedNmadd => sink.f64x2_relaxed_nmadd(),
            Instruction::I8x16RelaxedLaneselect => sink.i8x16_relaxed_laneselect(),
            Instruction::I16x8RelaxedLaneselect => sink.i16x8_relaxed_laneselect(),
            Instruction::I32x4RelaxedLaneselect => sink.i32x4_relaxed_laneselect(),
            Instruction::I64x2RelaxedLaneselect => sink.i64x2_relaxed_laneselect(),
            Instruction::F32x4RelaxedMin => sink.f32x4_relaxed_min(),
            Instruction::F32x4RelaxedMax => sink.f32x4_relaxed_max(),
            Instruction::F64x2RelaxedMin => sink.f64x2_relaxed_min(),
            Instruction::F64x2RelaxedMax => sink.f64x2_relaxed_max(),
            Instruction::I16x8RelaxedQ15mulrS => sink.i16x8_relaxed_q15mulr_s(),
            Instruction::I16x8RelaxedDotI8x16I7x16S => sink.i16x8_relaxed_dot_i8x16_i7x16_s(),
            Instruction::I32x4RelaxedDotI8x16I7x16AddS => {
                sink.i32x4_relaxed_dot_i8x16_i7x16_add_s()
            }

            // Atomic instructions from the thread proposal
            Instruction::MemoryAtomicNotify(memarg) => sink.memory_atomic_notify(memarg),
            Instruction::MemoryAtomicWait32(memarg) => sink.memory_atomic_wait32(memarg),
            Instruction::MemoryAtomicWait64(memarg) => sink.memory_atomic_wait64(memarg),
            Instruction::AtomicFence => sink.atomic_fence(),
            Instruction::I32AtomicLoad(memarg) => sink.i32_atomic_load(memarg),
            Instruction::I64AtomicLoad(memarg) => sink.i64_atomic_load(memarg),
            Instruction::I32AtomicLoad8U(memarg) => sink.i32_atomic_load8_u(memarg),
            Instruction::I32AtomicLoad16U(memarg) => sink.i32_atomic_load16_u(memarg),
            Instruction::I64AtomicLoad8U(memarg) => sink.i64_atomic_load8_u(memarg),
            Instruction::I64AtomicLoad16U(memarg) => sink.i64_atomic_load16_u(memarg),
            Instruction::I64AtomicLoad32U(memarg) => sink.i64_atomic_load32_u(memarg),
            Instruction::I32AtomicStore(memarg) => sink.i32_atomic_store(memarg),
            Instruction::I64AtomicStore(memarg) => sink.i64_atomic_store(memarg),
            Instruction::I32AtomicStore8(memarg) => sink.i32_atomic_store8(memarg),
            Instruction::I32AtomicStore16(memarg) => sink.i32_atomic_store16(memarg),
            Instruction::I64AtomicStore8(memarg) => sink.i64_atomic_store8(memarg),
            Instruction::I64AtomicStore16(memarg) => sink.i64_atomic_store16(memarg),
            Instruction::I64AtomicStore32(memarg) => sink.i64_atomic_store32(memarg),
            Instruction::I32AtomicRmwAdd(memarg) => sink.i32_atomic_rmw_add(memarg),
            Instruction::I64AtomicRmwAdd(memarg) => sink.i64_atomic_rmw_add(memarg),
            Instruction::I32AtomicRmw8AddU(memarg) => sink.i32_atomic_rmw8_add_u(memarg),
            Instruction::I32AtomicRmw16AddU(memarg) => sink.i32_atomic_rmw16_add_u(memarg),
            Instruction::I64AtomicRmw8AddU(memarg) => sink.i64_atomic_rmw8_add_u(memarg),
            Instruction::I64AtomicRmw16AddU(memarg) => sink.i64_atomic_rmw16_add_u(memarg),
            Instruction::I64AtomicRmw32AddU(memarg) => sink.i64_atomic_rmw32_add_u(memarg),
            Instruction::I32AtomicRmwSub(memarg) => sink.i32_atomic_rmw_sub(memarg),
            Instruction::I64AtomicRmwSub(memarg) => sink.i64_atomic_rmw_sub(memarg),
            Instruction::I32AtomicRmw8SubU(memarg) => sink.i32_atomic_rmw8_sub_u(memarg),
            Instruction::I32AtomicRmw16SubU(memarg) => sink.i32_atomic_rmw16_sub_u(memarg),
            Instruction::I64AtomicRmw8SubU(memarg) => sink.i64_atomic_rmw8_sub_u(memarg),
            Instruction::I64AtomicRmw16SubU(memarg) => sink.i64_atomic_rmw16_sub_u(memarg),
            Instruction::I64AtomicRmw32SubU(memarg) => sink.i64_atomic_rmw32_sub_u(memarg),
            Instruction::I32AtomicRmwAnd(memarg) => sink.i32_atomic_rmw_and(memarg),
            Instruction::I64AtomicRmwAnd(memarg) => sink.i64_atomic_rmw_and(memarg),
            Instruction::I32AtomicRmw8AndU(memarg) => sink.i32_atomic_rmw8_and_u(memarg),
            Instruction::I32AtomicRmw16AndU(memarg) => sink.i32_atomic_rmw16_and_u(memarg),
            Instruction::I64AtomicRmw8AndU(memarg) => sink.i64_atomic_rmw8_and_u(memarg),
            Instruction::I64AtomicRmw16AndU(memarg) => sink.i64_atomic_rmw16_and_u(memarg),
            Instruction::I64AtomicRmw32AndU(memarg) => sink.i64_atomic_rmw32_and_u(memarg),
            Instruction::I32AtomicRmwOr(memarg) => sink.i32_atomic_rmw_or(memarg),
            Instruction::I64AtomicRmwOr(memarg) => sink.i64_atomic_rmw_or(memarg),
            Instruction::I32AtomicRmw8OrU(memarg) => sink.i32_atomic_rmw8_or_u(memarg),
            Instruction::I32AtomicRmw16OrU(memarg) => sink.i32_atomic_rmw16_or_u(memarg),
            Instruction::I64AtomicRmw8OrU(memarg) => sink.i64_atomic_rmw8_or_u(memarg),
            Instruction::I64AtomicRmw16OrU(memarg) => sink.i64_atomic_rmw16_or_u(memarg),
            Instruction::I64AtomicRmw32OrU(memarg) => sink.i64_atomic_rmw32_or_u(memarg),
            Instruction::I32AtomicRmwXor(memarg) => sink.i32_atomic_rmw_xor(memarg),
            Instruction::I64AtomicRmwXor(memarg) => sink.i64_atomic_rmw_xor(memarg),
            Instruction::I32AtomicRmw8XorU(memarg) => sink.i32_atomic_rmw8_xor_u(memarg),
            Instruction::I32AtomicRmw16XorU(memarg) => sink.i32_atomic_rmw16_xor_u(memarg),
            Instruction::I64AtomicRmw8XorU(memarg) => sink.i64_atomic_rmw8_xor_u(memarg),
            Instruction::I64AtomicRmw16XorU(memarg) => sink.i64_atomic_rmw16_xor_u(memarg),
            Instruction::I64AtomicRmw32XorU(memarg) => sink.i64_atomic_rmw32_xor_u(memarg),
            Instruction::I32AtomicRmwXchg(memarg) => sink.i32_atomic_rmw_xchg(memarg),
            Instruction::I64AtomicRmwXchg(memarg) => sink.i64_atomic_rmw_xchg(memarg),
            Instruction::I32AtomicRmw8XchgU(memarg) => sink.i32_atomic_rmw8_xchg_u(memarg),
            Instruction::I32AtomicRmw16XchgU(memarg) => sink.i32_atomic_rmw16_xchg_u(memarg),
            Instruction::I64AtomicRmw8XchgU(memarg) => sink.i64_atomic_rmw8_xchg_u(memarg),
            Instruction::I64AtomicRmw16XchgU(memarg) => sink.i64_atomic_rmw16_xchg_u(memarg),
            Instruction::I64AtomicRmw32XchgU(memarg) => sink.i64_atomic_rmw32_xchg_u(memarg),
            Instruction::I32AtomicRmwCmpxchg(memarg) => sink.i32_atomic_rmw_cmpxchg(memarg),
            Instruction::I64AtomicRmwCmpxchg(memarg) => sink.i64_atomic_rmw_cmpxchg(memarg),
            Instruction::I32AtomicRmw8CmpxchgU(memarg) => sink.i32_atomic_rmw8_cmpxchg_u(memarg),
            Instruction::I32AtomicRmw16CmpxchgU(memarg) => sink.i32_atomic_rmw16_cmpxchg_u(memarg),
            Instruction::I64AtomicRmw8CmpxchgU(memarg) => sink.i64_atomic_rmw8_cmpxchg_u(memarg),
            Instruction::I64AtomicRmw16CmpxchgU(memarg) => sink.i64_atomic_rmw16_cmpxchg_u(memarg),
            Instruction::I64AtomicRmw32CmpxchgU(memarg) => sink.i64_atomic_rmw32_cmpxchg_u(memarg),

            // Atomic instructions from the shared-everything-threads proposal
            Instruction::GlobalAtomicGet {
                ordering,
                global_index,
            } => sink.global_atomic_get(ordering, global_index),
            Instruction::GlobalAtomicSet {
                ordering,
                global_index,
            } => sink.global_atomic_set(ordering, global_index),
            Instruction::GlobalAtomicRmwAdd {
                ordering,
                global_index,
            } => sink.global_atomic_rmw_add(ordering, global_index),
            Instruction::GlobalAtomicRmwSub {
                ordering,
                global_index,
            } => sink.global_atomic_rmw_sub(ordering, global_index),
            Instruction::GlobalAtomicRmwAnd {
                ordering,
                global_index,
            } => sink.global_atomic_rmw_and(ordering, global_index),
            Instruction::GlobalAtomicRmwOr {
                ordering,
                global_index,
            } => sink.global_atomic_rmw_or(ordering, global_index),
            Instruction::GlobalAtomicRmwXor {
                ordering,
                global_index,
            } => sink.global_atomic_rmw_xor(ordering, global_index),
            Instruction::GlobalAtomicRmwXchg {
                ordering,
                global_index,
            } => sink.global_atomic_rmw_xchg(ordering, global_index),
            Instruction::GlobalAtomicRmwCmpxchg {
                ordering,
                global_index,
            } => sink.global_atomic_rmw_cmpxchg(ordering, global_index),
            Instruction::TableAtomicGet {
                ordering,
                table_index,
            } => sink.table_atomic_get(ordering, table_index),
            Instruction::TableAtomicSet {
                ordering,
                table_index,
            } => sink.table_atomic_set(ordering, table_index),
            Instruction::TableAtomicRmwXchg {
                ordering,
                table_index,
            } => sink.table_atomic_rmw_xchg(ordering, table_index),
            Instruction::TableAtomicRmwCmpxchg {
                ordering,
                table_index,
            } => sink.table_atomic_rmw_cmpxchg(ordering, table_index),
            Instruction::StructAtomicGet {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_get(ordering, struct_type_index, field_index),
            Instruction::StructAtomicGetS {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_get_s(ordering, struct_type_index, field_index),
            Instruction::StructAtomicGetU {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_get_u(ordering, struct_type_index, field_index),
            Instruction::StructAtomicSet {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_set(ordering, struct_type_index, field_index),
            Instruction::StructAtomicRmwAdd {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_rmw_add(ordering, struct_type_index, field_index),
            Instruction::StructAtomicRmwSub {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_rmw_sub(ordering, struct_type_index, field_index),
            Instruction::StructAtomicRmwAnd {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_rmw_and(ordering, struct_type_index, field_index),
            Instruction::StructAtomicRmwOr {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_rmw_or(ordering, struct_type_index, field_index),
            Instruction::StructAtomicRmwXor {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_rmw_xor(ordering, struct_type_index, field_index),
            Instruction::StructAtomicRmwXchg {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_rmw_xchg(ordering, struct_type_index, field_index),
            Instruction::StructAtomicRmwCmpxchg {
                ordering,
                struct_type_index,
                field_index,
            } => sink.struct_atomic_rmw_cmpxchg(ordering, struct_type_index, field_index),
            Instruction::ArrayAtomicGet {
                ordering,
                array_type_index,
            } => sink.array_atomic_get(ordering, array_type_index),
            Instruction::ArrayAtomicGetS {
                ordering,
                array_type_index,
            } => sink.array_atomic_get_s(ordering, array_type_index),
            Instruction::ArrayAtomicGetU {
                ordering,
                array_type_index,
            } => sink.array_atomic_get_u(ordering, array_type_index),
            Instruction::ArrayAtomicSet {
                ordering,
                array_type_index,
            } => sink.array_atomic_set(ordering, array_type_index),
            Instruction::ArrayAtomicRmwAdd {
                ordering,
                array_type_index,
            } => sink.array_atomic_rmw_add(ordering, array_type_index),
            Instruction::ArrayAtomicRmwSub {
                ordering,
                array_type_index,
            } => sink.array_atomic_rmw_sub(ordering, array_type_index),
            Instruction::ArrayAtomicRmwAnd {
                ordering,
                array_type_index,
            } => sink.array_atomic_rmw_and(ordering, array_type_index),
            Instruction::ArrayAtomicRmwOr {
                ordering,
                array_type_index,
            } => sink.array_atomic_rmw_or(ordering, array_type_index),
            Instruction::ArrayAtomicRmwXor {
                ordering,
                array_type_index,
            } => sink.array_atomic_rmw_xor(ordering, array_type_index),
            Instruction::ArrayAtomicRmwXchg {
                ordering,
                array_type_index,
            } => sink.array_atomic_rmw_xchg(ordering, array_type_index),
            Instruction::ArrayAtomicRmwCmpxchg {
                ordering,
                array_type_index,
            } => sink.array_atomic_rmw_cmpxchg(ordering, array_type_index),
            Instruction::RefI31Shared => sink.ref_i31_shared(),
            Instruction::ContNew(type_index) => sink.cont_new(type_index),
            Instruction::ContBind {
                argument_index,
                result_index,
            } => sink.cont_bind(argument_index, result_index),
            Instruction::Suspend(tag_index) => sink.suspend(tag_index),
            Instruction::Resume {
                cont_type_index,
                ref resume_table,
            } => sink.resume(cont_type_index, resume_table.iter().cloned()),
            Instruction::ResumeThrow {
                cont_type_index,
                tag_index,
                ref resume_table,
            } => sink.resume_throw(cont_type_index, tag_index, resume_table.iter().cloned()),
            Instruction::Switch {
                cont_type_index,
                tag_index,
            } => sink.switch(cont_type_index, tag_index),
            Instruction::I64Add128 => sink.i64_add128(),
            Instruction::I64Sub128 => sink.i64_sub128(),
            Instruction::I64MulWideS => sink.i64_mul_wide_s(),
            Instruction::I64MulWideU => sink.i64_mul_wide_u(),
        };
    }
}

#[derive(Clone, Debug, serde_derive::Serialize, serde_derive::Deserialize)]
#[allow(missing_docs)]
pub enum Catch {
    One { tag: u32, label: u32 },
    OneRef { tag: u32, label: u32 },
    All { label: u32 },
    AllRef { label: u32 },
}

impl Encode for Catch {
    fn encode(&self, sink: &mut Vec<u8>) {
        match self {
            Catch::One { tag, label } => {
                sink.push(0x00);
                tag.encode(sink);
                label.encode(sink);
            }
            Catch::OneRef { tag, label } => {
                sink.push(0x01);
                tag.encode(sink);
                label.encode(sink);
            }
            Catch::All { label } => {
                sink.push(0x02);
                label.encode(sink);
            }
            Catch::AllRef { label } => {
                sink.push(0x03);
                label.encode(sink);
            }
        }
    }
}

#[derive(Clone, Debug, serde_derive::Serialize, serde_derive::Deserialize)]
#[allow(missing_docs)]
pub enum Handle {
    OnLabel { tag: u32, label: u32 },
    OnSwitch { tag: u32 },
}

impl Encode for Handle {
    fn encode(&self, sink: &mut Vec<u8>) {
        match self {
            Handle::OnLabel { tag, label } => {
                sink.push(0x00);
                tag.encode(sink);
                label.encode(sink);
            }
            Handle::OnSwitch { tag } => {
                sink.push(0x01);
                tag.encode(sink);
            }
        }
    }
}

/// A constant expression.
///
/// Usable in contexts such as offsets or initializers.
#[derive(Clone, Debug, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ConstExpr {
    bytes: Vec<u8>,
}

impl ConstExpr {
    /// Create a new empty constant expression builder.
    pub fn empty() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Create a constant expression with the specified raw encoding of instructions.
    pub fn raw(bytes: impl IntoIterator<Item = u8>) -> Self {
        Self {
            bytes: bytes.into_iter().collect(),
        }
    }

    /// Create a constant expression with the sequence of instructions
    pub fn extended<'a>(insns: impl IntoIterator<Item = Instruction<'a>>) -> Self {
        let mut bytes = vec![];
        for insn in insns {
            insn.encode(&mut bytes);
        }
        Self { bytes }
    }

    fn new<F>(f: F) -> Self
    where
        for<'a, 'b> F: FnOnce(&'a mut InstructionSink<'b>) -> &'a mut InstructionSink<'b>,
    {
        let mut bytes = vec![];
        f(&mut InstructionSink::new(&mut bytes));
        Self { bytes }
    }

    fn with<F>(mut self, f: F) -> Self
    where
        for<'a, 'b> F: FnOnce(&'a mut InstructionSink<'b>) -> &'a mut InstructionSink<'b>,
    {
        f(&mut InstructionSink::new(&mut self.bytes));
        self
    }

    /// Create a constant expression containing a single `global.get` instruction.
    pub fn global_get(index: u32) -> Self {
        Self::new(|insn| insn.global_get(index))
    }

    /// Create a constant expression containing a single `ref.null` instruction.
    pub fn ref_null(ty: HeapType) -> Self {
        Self::new(|insn| insn.ref_null(ty))
    }

    /// Create a constant expression containing a single `ref.func` instruction.
    pub fn ref_func(func: u32) -> Self {
        Self::new(|insn| insn.ref_func(func))
    }

    /// Create a constant expression containing a single `i32.const` instruction.
    pub fn i32_const(value: i32) -> Self {
        Self::new(|insn| insn.i32_const(value))
    }

    /// Create a constant expression containing a single `i64.const` instruction.
    pub fn i64_const(value: i64) -> Self {
        Self::new(|insn| insn.i64_const(value))
    }

    /// Create a constant expression containing a single `f32.const` instruction.
    pub fn f32_const(value: f32) -> Self {
        Self::new(|insn| insn.f32_const(value))
    }

    /// Create a constant expression containing a single `f64.const` instruction.
    pub fn f64_const(value: f64) -> Self {
        Self::new(|insn| insn.f64_const(value))
    }

    /// Create a constant expression containing a single `v128.const` instruction.
    pub fn v128_const(value: i128) -> Self {
        Self::new(|insn| insn.v128_const(value))
    }

    /// Add a `global.get` instruction to this constant expression.
    pub fn with_global_get(self, index: u32) -> Self {
        self.with(|insn| insn.global_get(index))
    }

    /// Add a `ref.null` instruction to this constant expression.
    pub fn with_ref_null(self, ty: HeapType) -> Self {
        self.with(|insn| insn.ref_null(ty))
    }

    /// Add a `ref.func` instruction to this constant expression.
    pub fn with_ref_func(self, func: u32) -> Self {
        self.with(|insn| insn.ref_func(func))
    }

    /// Add an `i32.const` instruction to this constant expression.
    pub fn with_i32_const(self, value: i32) -> Self {
        self.with(|insn| insn.i32_const(value))
    }

    /// Add an `i64.const` instruction to this constant expression.
    pub fn with_i64_const(self, value: i64) -> Self {
        self.with(|insn| insn.i64_const(value))
    }

    /// Add a `f32.const` instruction to this constant expression.
    pub fn with_f32_const(self, value: f32) -> Self {
        self.with(|insn| insn.f32_const(value))
    }

    /// Add a `f64.const` instruction to this constant expression.
    pub fn with_f64_const(self, value: f64) -> Self {
        self.with(|insn| insn.f64_const(value))
    }

    /// Add a `v128.const` instruction to this constant expression.
    pub fn with_v128_const(self, value: i128) -> Self {
        self.with(|insn| insn.v128_const(value))
    }

    /// Add an `i32.add` instruction to this constant expression.
    pub fn with_i32_add(self) -> Self {
        self.with(|insn| insn.i32_add())
    }

    /// Add an `i32.sub` instruction to this constant expression.
    pub fn with_i32_sub(self) -> Self {
        self.with(|insn| insn.i32_sub())
    }

    /// Add an `i32.mul` instruction to this constant expression.
    pub fn with_i32_mul(self) -> Self {
        self.with(|insn| insn.i32_mul())
    }

    /// Add an `i64.add` instruction to this constant expression.
    pub fn with_i64_add(self) -> Self {
        self.with(|insn| insn.i64_add())
    }

    /// Add an `i64.sub` instruction to this constant expression.
    pub fn with_i64_sub(self) -> Self {
        self.with(|insn| insn.i64_sub())
    }

    /// Add an `i64.mul` instruction to this constant expression.
    pub fn with_i64_mul(self) -> Self {
        self.with(|insn| insn.i64_mul())
    }

    /// Returns the function, if any, referenced by this global.
    pub fn get_ref_func(&self) -> Option<u32> {
        let prefix = *self.bytes.get(0)?;
        // 0xd2 == `ref.func` opcode, and if that's found then load the leb
        // corresponding to the function index.
        if prefix != 0xd2 {
            return None;
        }
        leb128fmt::decode_uint_slice::<u32, 32>(&self.bytes[1..], &mut 0).ok()
    }
}

impl Encode for ConstExpr {
    fn encode(&self, sink: &mut Vec<u8>) {
        sink.extend(&self.bytes);
        InstructionSink::new(sink).end();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn function_new_with_locals_test() {
        use super::*;

        // Test the algorithm for conversion is correct
        let f1 = Function::new_with_locals_types([
            ValType::I32,
            ValType::I32,
            ValType::I64,
            ValType::F32,
            ValType::F32,
            ValType::F32,
            ValType::I32,
            ValType::I64,
            ValType::I64,
        ]);
        let f2 = Function::new([
            (2, ValType::I32),
            (1, ValType::I64),
            (3, ValType::F32),
            (1, ValType::I32),
            (2, ValType::I64),
        ]);

        assert_eq!(f1.bytes, f2.bytes)
    }

    #[test]
    fn func_raw_bytes() {
        use super::*;

        let mut f = Function::new([(1, ValType::I32), (1, ValType::F32)]);
        f.instructions().end();
        let mut code_from_func = CodeSection::new();
        code_from_func.function(&f);
        let bytes = f.into_raw_body();
        let mut code_from_raw = CodeSection::new();
        code_from_raw.raw(&bytes[..]);

        let mut c1 = vec![];
        code_from_func.encode(&mut c1);
        let mut c2 = vec![];
        code_from_raw.encode(&mut c2);
        assert_eq!(c1, c2);
    }
}
