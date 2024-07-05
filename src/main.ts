type Pos = readonly [line: number, col: number];

interface Loc {
  readonly start: Pos;
  readonly end: Pos;
}

enum TokenKind {
  INT = "INT",
  SYMBOL = "SYMBOL",
  OPEN_PAREN = "OPEN_PAREN",
  CLOSE_PAREN = "CLOSE_PAREN",
  COMMA = "COMMA",
  WHITESPACE = "WHITESPACE",
  FN_KEYWORD = "FN_KEYWORD",
  LET_KEYWORD = "LET_KEYWORD",
  UNKNOWN = "UNKNOWN",
}

interface Token extends Loc {
  readonly kind: TokenKind;
  readonly lexeme: string;
}

class Lexer {
  readonly #code: string;

  #offset = 0;
  #line = 1;
  #col = 1;

  constructor(code: string) {
    this.#code = code;
  }

  #peek() {
    return this.#code[this.#offset] ?? "";
  }

  #pos(): Pos {
    return [this.#line, this.#col];
  }

  #next() {
    const char = this.#code[this.#offset++];
    switch (char) {
      case undefined:
        return "";

      case "\n":
        this.#line++;
        this.#col = 1;
        break;

      default:
        this.#col++;
        break;
    }

    return char;
  }

  get #isAtEnd() {
    return this.#offset >= this.#code.length;
  }

  *lex() {
    while (!this.#isAtEnd) {
      yield this.#nextToken();
    }
  }

  #nextToken(): Token {
    const nextChar = this.#peek();

    switch (nextChar) {
      case "(":
        return this.#singleChar(TokenKind.OPEN_PAREN);
      case ")":
        return this.#singleChar(TokenKind.CLOSE_PAREN);
      case ",":
        return this.#singleChar(TokenKind.COMMA);
    }

    if (/^\s$/.test(nextChar)) {
      return this.#whitespace();
    } else if (/^\d$/.test(nextChar)) {
      return this.#int();
    } else if (/^[a-z_]$/i.test(nextChar)) {
      return this.#symbolOrKeyword();
    } else {
      return this.#singleChar(TokenKind.UNKNOWN);
    }
  }

  #whitespace(): Token {
    const start = this.#pos();

    let lexeme = "";
    while (/^\s$/.test(this.#peek())) {
      lexeme += this.#next();
    }

    const end = this.#pos();

    return {
      kind: TokenKind.WHITESPACE,
      start,
      end,
      lexeme,
    };
  }

  #symbolOrKeyword(): Token {
    const start = this.#pos();
    const lexeme = this.#symbolLexeme();
    const end = this.#pos();

    let kind;
    switch (lexeme) {
      case "let":
        kind = TokenKind.LET_KEYWORD;
        break;

      case "fn":
        kind = TokenKind.FN_KEYWORD;
        break;

      default:
        kind = TokenKind.SYMBOL;
        break;
    }

    return {
      kind,
      lexeme,
      start,
      end,
    };
  }

  #symbolLexeme(): string {
    let lexeme = "";
    while (/^[a-z_0-9]$/i.test(this.#peek())) {
      lexeme += this.#next();
    }
    return lexeme;
  }

  #singleChar(kind: TokenKind): Token {
    const start = this.#pos();
    const lexeme = this.#next();
    const end = this.#pos();
    return { kind, lexeme, start, end };
  }

  #int(): Token {
    const start = this.#pos();

    let lexeme = "";
    while (/^\d$/.test(this.#peek())) {
      lexeme += this.#next();
    }

    const end = this.#pos();

    return {
      kind: TokenKind.INT,
      lexeme,
      start,
      end,
    };
  }
}

type Expression = Int | Identifier | Call | Fn | LetBinding;

interface Int extends Loc {
  kind: "int";
  int: number;
}

interface Identifier extends Loc {
  kind: "identifier";
  identifier: string;
}

interface Call extends Loc {
  kind: "call";
  callee: Expression;
  args: Expression[];
}

interface Fn extends Loc {
  kind: "fn";
  params: Identifier[];
  body: Expression;
}

interface LetBinding extends Loc {
  kind: "letbinding";
  variable: Identifier;
  binding: Expression;
  next: Expression;
}

enum DiagnosticKind {
  ERROR = "ERROR",
}

abstract class Diagnostic {
  abstract readonly kind: DiagnosticKind;
  abstract readonly loc: Loc;
  abstract readonly message: string;

  toString() {
    return `${this.loc.start.join(":")}: ${this.message}`;
  }
}

class UnexpectedEOF extends Diagnostic {
  readonly kind = DiagnosticKind.ERROR;

  readonly loc: Loc;

  constructor(loc: Loc) {
    super();
    this.loc = loc;
  }

  get message() {
    return "Unexpected end of file";
  }
}

class UnexpectedToken extends Diagnostic {
  readonly kind = DiagnosticKind.ERROR;

  readonly token: Token;
  readonly expectedOneOf: TokenKind[];

  constructor(token: Token, expectedOneOf: TokenKind[]) {
    super();
    this.token = token;
    this.expectedOneOf = expectedOneOf;
  }

  get loc() {
    return this.token;
  }

  get message() {
    const expectation =
      this.expectedOneOf.length === 1
        ? this.expectedOneOf[0]
        : `one of ${this.expectedOneOf.join(", ")}`;

    return `Unexpected token ${JSON.stringify(this.token.lexeme)} (${this.token.kind}), expected ${expectation}`;
  }
}

class UndefinedReference extends Diagnostic {
  readonly kind = DiagnosticKind.ERROR;

  readonly reference: Identifier;

  constructor(reference: Identifier) {
    super();
    this.reference = reference;
  }

  get loc() {
    return this.reference;
  }

  get message() {
    return `Undefined reference "${this.reference.identifier}"`;
  }
}

class TypeMismatch extends Diagnostic {
  readonly kind = DiagnosticKind.ERROR;

  readonly lhs: Type;
  readonly rhs: Type;

  readonly loc: Loc;

  constructor(lhs: Type, rhs: Type, loc: Loc) {
    super();
    this.lhs = lhs;
    this.rhs = rhs;
    this.loc = loc;
  }

  get message() {
    const typevars = new Map();
    return `Type mismatch: ${formatType(this.lhs, typevars)} != ${formatType(this.rhs, typevars)}`;
  }
}

function formatType(
  type: Type,
  namedTypeVars = new Map<TypeVar, string>(),
): string {
  switch (type.kind) {
    case "int":
      return "Int";
    case "var":
      const existingName = namedTypeVars.get(type);
      if (existingName) return existingName;
      const newName = String.fromCharCode(97 + namedTypeVars.size);
      namedTypeVars.set(type, newName);
      return newName;
    case "fn":
      return `(${type.parameters.map((t) => formatType(t, namedTypeVars))}) -> ${formatType(type.returnType, namedTypeVars)}`;
  }
}

class Diagnostics {
  readonly #diagnostics: Diagnostic[] = [];

  add(diagnostic: Diagnostic) {
    this.#diagnostics.push(diagnostic);
  }

  get isEmpty() {
    return this.#diagnostics.length === 0;
  }

  *[Symbol.iterator]() {
    yield* this.#diagnostics;
  }

  catch<R>(f: () => R): R | undefined {
    try {
      return f();
    } catch (e) {
      if (e instanceof Diagnostic) {
        this.add(e);
        return undefined;
      }
      throw e;
    }
  }
}

class Parser {
  readonly #tokens: Token[];
  readonly #diagnostics: Diagnostics;

  #lastLoc: Loc = { start: [1, 1], end: [1, 1] };

  constructor(tokens: Token[], diagnostics: Diagnostics) {
    this.#tokens = tokens.filter((t) => t.kind !== TokenKind.WHITESPACE);
    this.#diagnostics = diagnostics;
  }

  #peek(): Token | undefined {
    return this.#tokens[0];
  }

  #next(): Token {
    const token = this.#tokens.shift();
    if (!token) {
      throw new UnexpectedEOF(this.#lastLoc);
    }
    this.#lastLoc = token;
    return token;
  }

  parseExpression(): Expression {
    let expression = this.#parseTerm();
    while (this.#peek()?.kind === TokenKind.OPEN_PAREN) {
      expression = this.#parseCallOnCallee(expression);
    }
    return expression;
  }

  #parseTerm(): Expression {
    switch (
      this.#assertPeek(
        TokenKind.INT,
        TokenKind.SYMBOL,
        TokenKind.LET_KEYWORD,
        TokenKind.FN_KEYWORD,
      ).kind
    ) {
      case TokenKind.INT:
        return this.parseInt();

      case TokenKind.SYMBOL:
        return this.parseIdentifier();

      case TokenKind.LET_KEYWORD:
        return this.parseLetBinding();

      case TokenKind.FN_KEYWORD:
        return this.parseFn();
    }
  }

  #assertNext<const K extends TokenKind[]>(
    ...kinds: K
  ): Token & { kind: K[number] } {
    const nextToken = this.#next();
    if (!kinds.includes(nextToken.kind))
      throw new UnexpectedToken(nextToken, kinds);
    return nextToken;
  }

  #assertPeek<const K extends TokenKind[]>(
    ...kinds: K
  ): Token & { kind: K[number] } {
    const nextToken = this.#peek();
    if (!nextToken) {
      throw new UnexpectedEOF(this.#lastLoc);
    }
    if (!kinds.includes(nextToken.kind))
      throw new UnexpectedToken(nextToken, kinds);
    return nextToken;
  }

  parseInt(): Int {
    const token = this.#assertNext(TokenKind.INT);

    return {
      kind: "int",
      int: parseInt(token.lexeme),
      start: token.start,
      end: token.end,
    };
  }

  parseIdentifier(): Identifier {
    const token = this.#assertNext(TokenKind.SYMBOL);

    return {
      kind: "identifier",
      identifier: token.lexeme,
      start: token.start,
      end: token.end,
    };
  }

  parseCall(): Call {
    const callee = this.#parseTerm();
    return this.#parseCallOnCallee(callee);
  }

  #parseCommaSeparated<E>(until: TokenKind, f: () => E): E[] {
    const elements = [];

    nextElement: while (this.#peek()?.kind !== until) {
      try {
        elements.push(f());
      } catch (e) {
        if (!(e instanceof Diagnostic)) {
          throw e;
        }

        this.#diagnostics.add(e);
        while (true) {
          const next = this.#next();
          if (next.kind === until) return elements;
          if (next.kind === TokenKind.COMMA) continue nextElement;
        }
      }

      if (this.#peek()?.kind === until) break;

      this.#assertNext(TokenKind.COMMA);
    }

    return elements;
  }

  #parseCallOnCallee(callee: Expression): Call {
    this.#assertNext(TokenKind.OPEN_PAREN);
    const args = this.#parseCommaSeparated(TokenKind.CLOSE_PAREN, () =>
      this.parseExpression(),
    );
    const closeParen = this.#assertNext(TokenKind.CLOSE_PAREN);

    return {
      kind: "call",
      callee,
      args,
      start: callee.start,
      end: closeParen.end,
    };
  }

  parseFn(): Fn {
    const keyword = this.#assertNext(TokenKind.FN_KEYWORD);

    this.#assertNext(TokenKind.OPEN_PAREN);
    const params = this.#parseCommaSeparated(TokenKind.CLOSE_PAREN, () =>
      this.parseIdentifier(),
    );
    this.#assertNext(TokenKind.CLOSE_PAREN);

    this.#assertNext(TokenKind.OPEN_PAREN);
    const body = this.parseExpression();
    const closeParen = this.#assertNext(TokenKind.CLOSE_PAREN);

    return {
      kind: "fn",
      params,
      body,
      start: keyword.start,
      end: closeParen.end,
    };
  }

  #parseLet(): [Token, Identifier, Expression] {
    const keyword = this.#assertNext(TokenKind.LET_KEYWORD);

    const variable = this.parseIdentifier();
    const binding = this.parseExpression();

    return [keyword, variable, binding];
  }

  parseReplLine(): { alias?: Identifier; expression: Expression } {
    if (this.#peek()?.kind === TokenKind.LET_KEYWORD) {
      const [_, alias, expression] = this.#parseLet();
      return { alias, expression };
    }
    return { expression: this.parseExpression() };
  }

  parseLetBinding(): LetBinding {
    const [keyword, variable, binding] = this.#parseLet();
    const next = this.parseExpression();

    return {
      kind: "letbinding",
      variable,
      binding,
      next,
      start: keyword.start,
      end: next.end,
    };
  }
}

abstract class Visitor {
  visitExpression(node: Expression) {
    switch (node.kind) {
      case "int":
        return this.visitInt(node);
      case "identifier":
        return this.visitIdentifier(node);
      case "call":
        return this.visitCall(node);
      case "fn":
        return this.visitFn(node);
      case "letbinding":
        return this.visitLetBinding(node);
    }
  }

  visitInt(_node: Int) {}

  visitIdentifier(_node: Identifier) {}

  visitCall(node: Call) {
    this.visitExpression(node.callee);
    for (const arg of node.args) {
      this.visitExpression(arg);
    }
  }

  visitFn(node: Fn) {
    for (const param of node.params) {
      this.visitIdentifier(param);
    }
    this.visitExpression(node.body);
  }

  visitLetBinding(node: LetBinding) {
    this.visitIdentifier(node.variable);
    this.visitExpression(node.binding);
    this.visitExpression(node.next);
  }
}

interface Scope {
  parent?: Scope;
  declarations: Identifier[];
}

class SymbolTable extends Visitor {
  readonly #diagnostics: Diagnostics;

  readonly declarationsByReference = new Map<Identifier, Identifier>();

  #scope?: Scope;

  constructor(diagnostics: Diagnostics, rootScope?: Scope) {
    super();
    this.#diagnostics = diagnostics;
    this.#scope = rootScope;
  }

  #enterScope(declarations: Identifier[]) {
    this.#scope = { parent: this.#scope, declarations };
  }

  #leaveScope() {
    this.#scope = this.#scope?.parent;
  }

  visitLet(variable: Identifier, binding: Expression) {
    this.visitExpression(binding);
    this.#enterScope([variable]);
  }

  override visitLetBinding(node: LetBinding): void {
    this.visitLet(node.variable, node.binding);
    this.visitExpression(node.next);
    this.#leaveScope();
  }

  override visitFn(node: Fn): void {
    this.#enterScope(node.params);
    this.visitExpression(node.body);
    this.#leaveScope();
  }

  override visitIdentifier(reference: Identifier): void {
    let scope = this.#scope;
    while (scope) {
      const declaration = scope.declarations.find(
        (d) => d.identifier === reference.identifier,
      );
      if (!declaration) {
        scope = scope.parent;
        continue;
      }

      this.declarationsByReference.set(reference, declaration);
      return;
    }
    this.#diagnostics.add(new UndefinedReference(reference));
  }
}

type Type = IntType | FnType | TypeVar;

interface TypeVar {
  kind: "var";
  id: number;
}

let TYPE_VAR_COUNTER = 0;
function createTypeVar(): TypeVar {
  return { kind: "var", id: TYPE_VAR_COUNTER++ };
}

interface IntType {
  kind: "int";
}

interface FnType {
  kind: "fn";
  parameters: Type[];
  returnType: Type;
}

interface TypeEnvironment {
  symbolTable: SymbolTable;
  bindingTypes: Map<Identifier, Type>;
  boundTypeVars: Map<TypeVar, Type>;
}

class TypeChecker {
  readonly #diagnostics: Diagnostics;

  constructor(diagnostics: Diagnostics) {
    this.#diagnostics = diagnostics;
  }

  typeOfExpression(node: Expression, env: TypeEnvironment): Type {
    switch (node.kind) {
      case "int":
        return this.typeOfInt(node, env);
      case "identifier":
        return this.typeOfIdentifier(node, env);
      case "call":
        return this.typeOfCall(node, env);
      case "fn":
        return this.typeOfFn(node, env);
      case "letbinding":
        return this.typeOfLetBinding(node, env);
    }
  }

  typeOfInt(_node: Int, _env: TypeEnvironment): Type {
    return { kind: "int" };
  }

  typeOfIdentifier(node: Identifier, env: TypeEnvironment): Type {
    const declaration = env.symbolTable.declarationsByReference.get(node);
    if (declaration == null) return createTypeVar();

    return env.bindingTypes.get(declaration) ?? createTypeVar();
  }

  typeOfCall(node: Call, env: TypeEnvironment): Type {
    const callee = this.typeOfExpression(node.callee, env);
    const args = node.args.map((arg) => this.typeOfExpression(arg, env));

    const expectedType: FnType = {
      kind: "fn",
      parameters: args,
      returnType: createTypeVar(),
    };

    const [unifiedType] = this.#unify(node, expectedType, callee, env);

    return unifiedType.kind === "fn" ? unifiedType.returnType : createTypeVar();
  }

  #bindTypeVar(
    typevar: TypeVar,
    binding: Type,
    env: TypeEnvironment,
  ): [Type, TypeEnvironment] {
    if (binding.kind === "var") {
      binding = env.boundTypeVars.get(binding) ?? binding;
    }

    const boundTypeVars = new Map(env.boundTypeVars);
    boundTypeVars.set(typevar, binding);
    return [
      this.#replace(binding, typevar, binding),
      { ...env, boundTypeVars },
    ];
  }

  #unify(
    loc: Expression,
    lhs: Type,
    rhs: Type,
    env: TypeEnvironment,
  ): [Type, TypeEnvironment] {
    if (lhs.kind === "var") {
      const replacement = env.boundTypeVars.get(lhs);
      if (!replacement) {
        return this.#bindTypeVar(lhs, rhs, env);
      }
      lhs = replacement;
    }

    if (rhs.kind === "var") {
      const replacement = env.boundTypeVars.get(rhs);
      if (!replacement) {
        return this.#bindTypeVar(rhs, lhs, env);
      }
      rhs = replacement;
    }

    if (lhs.kind === "int" && rhs.kind === "int") {
      return [lhs, env];
    }

    if (
      lhs.kind === "fn" &&
      rhs.kind === "fn" &&
      lhs.parameters.length === rhs.parameters.length
    ) {
      const unifiedParams = [];

      for (let i = 0; i < lhs.parameters.length; i++) {
        const lhsParam = lhs.parameters[i];
        const rhsParam = rhs.parameters[i];

        let unifiedParam;
        [unifiedParam, env] = this.#unify(
          loc.kind === "call" ? loc.args[i] : loc,
          lhsParam,
          rhsParam,
          env,
        );

        unifiedParams.push(unifiedParam);
      }

      let unifiedReturnType;
      [unifiedReturnType, env] = this.#unify(
        loc,
        lhs.returnType,
        rhs.returnType,
        env,
      );

      return [
        {
          kind: "fn",
          parameters: unifiedParams,
          returnType: unifiedReturnType,
        },
        env,
      ];
    }

    this.#diagnostics.add(new TypeMismatch(lhs, rhs, loc));
    return [createTypeVar(), env];
  }

  #replace(subject: Type, typevar: TypeVar, replacement: Type): Type {
    switch (subject.kind) {
      case "var":
        if (subject.id === typevar.id) return replacement;
        return subject;
      case "int":
        return subject;
      case "fn":
        return {
          kind: "fn",
          parameters: subject.parameters.map((p) =>
            this.#replace(p, typevar, replacement),
          ),
          returnType: this.#replace(subject.returnType, typevar, replacement),
        };
    }
  }

  typeOfFn(node: Fn, env: TypeEnvironment): Type {
    const bindingTypes = new Map(env.bindingTypes);
    const parameters = [];

    for (const param of node.params) {
      const typevar = createTypeVar();
      bindingTypes.set(param, typevar);
      parameters.push(typevar);
    }

    env = { ...env, bindingTypes };

    const fnType: FnType = {
      kind: "fn",
      parameters,
      returnType: this.typeOfExpression(node.body, env),
    };

    return fnType;
  }

  typeOfLetBinding(node: LetBinding, env: TypeEnvironment): Type {
    const bindingTypes = new Map(env.bindingTypes);
    bindingTypes.set(node.variable, this.typeOfExpression(node.binding, env));

    env = { ...env, bindingTypes };

    return this.typeOfExpression(node.next, env);
  }
}

interface ExecutionContext {
  symbolTable: SymbolTable;
  bindings: Map<Identifier, Value>;
  callStack: Fn[];
}

type Value = IntValue | FnValue;

enum ValueTag {
  INT,
  FN,
}

interface IntValue {
  tag: ValueTag.INT;
  value: number;
}

interface FnValue {
  tag: ValueTag.FN;
  ctx: ExecutionContext;
  fn: Fn;
}

class RuntimeError extends Error {
  readonly callStack: Fn[];

  constructor(message: string, callStack: Fn[]) {
    super(message);

    this.callStack = callStack;
    this.stack =
      message +
      "\n" +
      callStack.map((frame) => "  " + frame.start.join(":")).join("\n");
  }
}

class Interpreter {
  evaluateExpression(node: Expression, ctx: ExecutionContext): Value {
    switch (node.kind) {
      case "int":
        return this.evaluateInt(node, ctx);
      case "identifier":
        return this.evaluateIdentifier(node, ctx);
      case "call":
        return this.evaluateCall(node, ctx);
      case "fn":
        return this.evaluateFn(node, ctx);
      case "letbinding":
        return this.evaluateLetBinding(node, ctx);
    }
  }

  evaluateInt(node: Int, _ctx: ExecutionContext): Value {
    return {
      tag: ValueTag.INT,
      value: node.int,
    };
  }

  evaluateIdentifier(node: Identifier, ctx: ExecutionContext): Value {
    const declaration = ctx.symbolTable.declarationsByReference.get(node);
    if (declaration == null)
      throw new RuntimeError("Undefined reference", ctx.callStack);

    const value = ctx.bindings.get(declaration);
    if (value == null)
      throw new RuntimeError("Uninitialized variable", ctx.callStack);

    return value;
  }

  evaluateCall(node: Call, ctx: ExecutionContext): Value {
    const callee = this.evaluateExpression(node.callee, ctx);
    const args = node.args.map((a) => this.evaluateExpression(a, ctx));

    if (callee.tag !== ValueTag.FN)
      throw new RuntimeError("Calling a non-function", ctx.callStack);

    const bindings = new Map(callee.ctx.bindings);
    for (let i = 0; i < args.length; i++) {
      bindings.set(callee.fn.params[i], args[i]);
    }

    const callStack = [...ctx.callStack, callee.fn];
    return this.evaluateExpression(callee.fn.body, {
      ...callee.ctx,
      bindings,
      callStack,
    });
  }

  evaluateFn(fn: Fn, ctx: ExecutionContext): Value {
    return {
      tag: ValueTag.FN,
      fn,
      ctx,
    };
  }

  evaluateLetBinding(node: LetBinding, ctx: ExecutionContext): Value {
    const bindings = new Map(ctx.bindings);
    bindings.set(node.variable, this.evaluateExpression(node.binding, ctx));

    return this.evaluateExpression(node.next, { ...ctx, bindings });
  }
}

function formatValue(value: Value): string {
  switch (value.tag) {
    case ValueTag.INT:
      return value.value.toString();
    case ValueTag.FN:
      return "<fn>";
  }
}

import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const rl = readline.createInterface({ input, output });

const rootDeclarations = [];
let rootBindingTypes = new Map<Identifier, Type>();
let rootBindings = new Map<Identifier, Value>();

const interpreter = new Interpreter();

while (true) {
  const prompt = await rl.question("> ");

  const diagnostics = new Diagnostics();
  const lexer = new Lexer(prompt);
  const tokens = Array.from(lexer.lex());
  const parser = new Parser(tokens, diagnostics);
  const line = diagnostics.catch(() => parser.parseReplLine());
  if (!diagnostics.isEmpty) {
    for (const diagnostic of diagnostics) {
      console.log(diagnostic.toString());
    }
  }
  if (line == null) continue;

  const symbolTable = new SymbolTable(diagnostics, {
    declarations: rootDeclarations,
  });
  symbolTable.visitExpression(line.expression);

  if (!diagnostics.isEmpty) {
    for (const diagnostic of diagnostics) {
      console.log(diagnostic.toString());
    }
    continue;
  }

  if (line.alias) {
    rootDeclarations.push(line.alias);
  }

  const typeChecker = new TypeChecker(diagnostics);
  const astType = typeChecker.typeOfExpression(line.expression, {
    bindingTypes: rootBindingTypes,
    boundTypeVars: new Map(),
    symbolTable,
  });

  if (!diagnostics.isEmpty) {
    for (const diagnostic of diagnostics) {
      console.log(diagnostic.toString());
    }
    continue;
  }

  if (line.alias) {
    rootBindingTypes.set(line.alias, astType);
  }

  const result = interpreter.evaluateExpression(line.expression, {
    symbolTable,
    callStack: [],
    bindings: rootBindings,
  });

  if (line.alias) {
    rootBindings.set(line.alias, result);
  }

  console.log(formatValue(result), formatType(astType));
}
