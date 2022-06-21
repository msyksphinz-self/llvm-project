#include "Dialect.h"
#include "Parser.h"
#include "MLIRGen.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mysv;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { MYSV, MLIR };
} // namespace
namespace {
enum Action { None, DumpAST, DumpMLIR };
} // namespace

static cl::opt<enum Action>
emitAction("emit", cl::desc("Select the kind of output desired"),
           cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
           cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

/// Returns a MYSV AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<mysv::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}


int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::mysv::MYSVDialect>();

  // Handle '.mysv' input to the compiler.
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 6;
  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
  if (!module)
    return 1;

  module->dump();

  return 0;
}


int dumpAST() {
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}




int main(int argc, char **argv)
{
  cl::ParseCommandLineOptions(argc, argv, "mysv compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    dumpAST();
    return 0;
  case Action::DumpMLIR:
    dumpMLIR();
    return 0;
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;

}
