#include "AST.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mysv;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
 public:
  void dump(ModuleAST *node);

 private:
  void dump(AssignExprAST *assignExpr);
  void dump(ExprAST *expr);
  void dump(NumberExprAST *num);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};

} // namespace


/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *expr) {
  llvm::TypeSwitch<ExprAST *>(expr)
      .Case<NumberExprAST, AssignExprAST>(
          [&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Expr>\n";
      });
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(AssignExprAST *varDecl) {
  INDENT();
  llvm::errs() << "assign " << varDecl->getName();
  llvm::errs() << " " << loc(varDecl) << "\n";
  dump(varDecl->getInitVal());
}


/// A literal number, just print the value.
void ASTDumper::dump(NumberExprAST *num) {
  INDENT();
  llvm::errs() << num->getValue() << " " << loc(num) << "\n";
}


/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &f : *node)
    dump(&f);
}


namespace mysv {

// Public API
void dump(ModuleAST &module) { ASTDumper().dump(&module); }

} // namespace mysv
