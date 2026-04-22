# CHANGELOG

<!-- version list -->

## v1.3.6 (2026-04-22)

### Bug Fixes

- Mcp oauth discovery, implementation of 2025-03-26 spec flow compliance.
  ([`195116a`](https://github.com/gadz82/orchid/commit/195116af684ae6b7cf4ec12eecb3427ad0d0bb95))


## v1.3.5 (2026-04-22)

### Bug Fixes

- Mcp oauth management fixes to support http mcp oauth configuration.
  ([`1cc8752`](https://github.com/gadz82/orchid/commit/1cc87521d695b62350cc4bb877943c714ff2a00d))


## v1.3.4 (2026-04-22)

### Bug Fixes

- Batch embedding limits for main providers.
  ([`7f143ef`](https://github.com/gadz82/orchid/commit/7f143ef15016debde6debd90d0abeb698731e4c0))


## v1.3.3 (2026-04-21)

### Bug Fixes

- Missing mcp_token_store property accessor
  ([`e12413d`](https://github.com/gadz82/orchid/commit/e12413da193b3ed5410da3e9c1355385e879fdb9))


## v1.3.2 (2026-04-21)

### Bug Fixes

- Add beta disclaimer to highlight work-in-progress status and fixing pipeline pypa behaviour on
  release phase.
  ([`8a18b9a`](https://github.com/gadz82/orchid/commit/8a18b9a4cee81666abe02d67e5c588712f039f0d))


## v1.3.1 (2026-04-20)

### Bug Fixes

- **persistence**: Enable integrator-defined migrations support
  ([`886498f`](https://github.com/gadz82/orchid/commit/886498f688fffe5cc6fb4fbef97803e264219299))


## v1.3.0 (2026-04-17)

### Bug Fixes

- Improve structured output routing for small models
  ([`ca5f591`](https://github.com/gadz82/orchid/commit/ca5f5919efbfee263ce0e6e62ecd5bf68b4348e7))

### Continuous Integration

- Grant pull-requests: write permission to the test job
  ([`d097460`](https://github.com/gadz82/orchid/commit/d097460c793ba27e296560c3a29c0d7f650f2bcd))

### Features

- Add configurable LLM fallback models
  ([`5eccf9c`](https://github.com/gadz82/orchid/commit/5eccf9cfc451c4d164af973938980a4ed746a90f))

- Add HITL tool approval and LangChain tool wrappers
  ([`6e09f6b`](https://github.com/gadz82/orchid/commit/6e09f6bc540891fb530ea30ea9b2f3be82ccc452))

- Add LangGraph checkpointer integration for state persistence
  ([`18cc43c`](https://github.com/gadz82/orchid/commit/18cc43cfc2d6493b9101a1246175408cfef8c8ba))

- Add LLM retry_attempts for transient error handling
  ([`735c410`](https://github.com/gadz82/orchid/commit/735c410ee08f6313fcdd9dce7b1e7c619bf87709))

- Add multi-query RAG and parent-child chunking
  ([`eccfd78`](https://github.com/gadz82/orchid/commit/eccfd78f56fbbd62bbf832ef90b6b13e75136ecb))

- Add OrchidMetricsHandler for observability
  ([`2bd283a`](https://github.com/gadz82/orchid/commit/2bd283a5bb813de4c6ffc4d406efc430a410fa4c))

- Add query reformulation for better RAG and tool search
  ([`58076d4`](https://github.com/gadz82/orchid/commit/58076d45ed9a60808dfbe6bd7a2c5444f5308f5b))

- Add resume endpoint to support HITL tool approval
  ([`2aec667`](https://github.com/gadz82/orchid/commit/2aec6670d1c95a414286cf78dd2685ca866f7eaf))

- Add shared hooks and utilities for CLI and frontend
  ([`d587e1e`](https://github.com/gadz82/orchid/commit/d587e1e4cff2cff256a8eb37fc752079c03a7ab7))

- Add streaming_enabled config to SupervisorConfig
  ([`c08b562`](https://github.com/gadz82/orchid/commit/c08b562c53bb0eadd357f88315dbe10a7a163fc8))

- Enable LLM response caching and add configuration tests
  ([`27835f2`](https://github.com/gadz82/orchid/commit/27835f26567a537940a95f322a74829209592243))

- Remove dead LLMProvider and LiteLLMProvider files
  ([`8585799`](https://github.com/gadz82/orchid/commit/8585799b2c9f1397ecd83284cc7ca935a1db92f5))

- Replace custom chunker with LangChain splitter
  ([`c06fc16`](https://github.com/gadz82/orchid/commit/c06fc1684782fd4aac97320d52e4c03930c042ef))

- Replace Embedder ABC with LangChain Embeddings
  ([`efaeaf6`](https://github.com/gadz82/orchid/commit/efaeaf63707a2a2ca5960561ceed4e5b607a654a))

- Replace LLMProvider with LangChain BaseChatModel
  ([`b489c78`](https://github.com/gadz82/orchid/commit/b489c78e8f231f4a67bc5b9aa3f2ee890e1995cc))

- Replace Orchid Document with LangChain Document
  ([`4154364`](https://github.com/gadz82/orchid/commit/4154364b59fd95236503767804f959e6025c0fa6))

- Update AGENTS.md docs for LangChain integration
  ([`0bec1d0`](https://github.com/gadz82/orchid/commit/0bec1d0b6fef09ddbaf180f4f16e747924369010))

- Use structured output for supervisor routing decisions
  ([`7b61fc1`](https://github.com/gadz82/orchid/commit/7b61fc12d4f3e755d92e5a68f2f55540d1bf98ad))

- **client**: Add OrchidClient with invoke/resume/stream
  ([`bc5b268`](https://github.com/gadz82/orchid/commit/bc5b268c006d3c6fc9f8c9a85ce1ce5f7f2f8af6))

### Refactoring

- Centralize helpers and streamline PKCE OAuth flow
  ([`a39798b`](https://github.com/gadz82/orchid/commit/a39798b9a2f5bce8b2f727423a186301c546e4fd))

- Delegate GenericAgent tool loop to AgenticLoop
  ([`bbf6cc1`](https://github.com/gadz82/orchid/commit/bbf6cc1b4aa18d0ad26deabbdfe9884fa7f5a8e0))

- Improve history handling, timeouts, and RAG robustness
  ([`b1dba20`](https://github.com/gadz82/orchid/commit/b1dba20836cb281b1c6e9c3bc844c054039fba86))

- Streamline client/bootstrap and expand test coverage
  ([`683c4c0`](https://github.com/gadz82/orchid/commit/683c4c07f60140aca57af8403e39c2eca6f18c2c))

### Testing

- Add HITL tool approval tests
  ([`2aec667`](https://github.com/gadz82/orchid/commit/2aec6670d1c95a414286cf78dd2685ca866f7eaf))


## v1.2.14 (2026-04-15)

### Bug Fixes

- Add MCPTokenStore implementation for OAuth token persistence
  ([`8938d33`](https://github.com/gadz82/orchid/commit/8938d336cfbe9d6109e772c45de11a4e07ffad8b))

- Add support for per-server MCP authentication configuration
  ([`87c0bb9`](https://github.com/gadz82/orchid/commit/87c0bb900ab72a308d67aa88e26f5e4d68950692))

- Add support for per-server MCP authentication configuration
  ([`15d1ca2`](https://github.com/gadz82/orchid/commit/15d1ca2da18d62271823a79c76ad59af2e457f86))

- Handle MCPAuthRequiredError for improved OAuth error isolation
  ([`3171508`](https://github.com/gadz82/orchid/commit/3171508c258548717e5ac85a76fc407fff16b8aa))

- Inject MCP auth status into routing logic for improved agent routing decisions
  ([`33b424b`](https://github.com/gadz82/orchid/commit/33b424bafaf8accb364ee4653ecf496797eb4bf0))

- Introduce MCPAuthRegistry for OAuth server management
  ([`e9bb5f4`](https://github.com/gadz82/orchid/commit/e9bb5f4809f578a4c563643b68f840e635440456))

- Managed per-server OAuth token management and tests
  ([`62ebba2`](https://github.com/gadz82/orchid/commit/62ebba2613dc3081f238c3b46a613b8a13b74521))


## v1.2.13 (2026-04-14)

### Bug Fixes

- Remove deprecated backward compatibility for runtime and import utilities
  ([`b139f61`](https://github.com/gadz82/orchid/commit/b139f610e6da7d1767dad1710042ac168ad88f43))


## v1.2.12 (2026-04-14)

### Bug Fixes

- Enhance error handling across MCP tools and servers with graceful degradation
  ([`f66d204`](https://github.com/gadz82/orchid/commit/f66d20477277f552f0e09a7285b4d6ae5c03d93a))


## v1.2.11 (2026-04-14)

### Bug Fixes

- Restructure GenericAgent tool-calling pipeline to enhance agentic loop consolidation
  ([`c0188a3`](https://github.com/gadz82/orchid/commit/c0188a3714e000bd5c18369285d1ebbe9a99146f))


## v1.2.10 (2026-04-14)

### Bug Fixes

- Implement multi-turn LLM tool loop with max turn limit and enhanced test coverage
  ([`9f57288`](https://github.com/gadz82/orchid/commit/9f57288447d70a5317fe9de0070fb7670b1d7400))


## v1.2.9 (2026-04-14)

### Bug Fixes

- Improve LLM-driven tool execution with parameter handling and test coverage
  ([`e7cac72`](https://github.com/gadz82/orchid/commit/e7cac7254949b04e7dbbd7cafd0edb80949c2263))


## v1.2.8 (2026-04-14)

### Bug Fixes

- Built-in tools args propagation fix.
  ([`6cec1d5`](https://github.com/gadz82/orchid/commit/6cec1d5184a8df8fc12c23ec129469d4b3835078))


## v1.2.7 (2026-04-14)

### Bug Fixes

- Built-in tools args propagation fix.
  ([`b6b0db0`](https://github.com/gadz82/orchid/commit/b6b0db0a7bf36b404fa96e9f0e1ce0e3a2f8672a))


## v1.2.6 (2026-04-14)

### Bug Fixes

- Built-in tools auth context propagation.
  ([`9633a12`](https://github.com/gadz82/orchid/commit/9633a12155a66a01c69063fb094b4d4b348edf58))


## v1.2.5 (2026-04-14)

### Bug Fixes

- Built-in tools parameter declarations in config.
  ([`b8dc60f`](https://github.com/gadz82/orchid/commit/b8dc60f226ffb3b2e637af8501b25ef89e284ff3))


## v1.2.4 (2026-04-14)

### Bug Fixes

- Prompt chain optimization and configuration parameters.
  ([`ef72430`](https://github.com/gadz82/orchid/commit/ef7243058f7a3e021c81cb5774037848ad057751))


## v1.2.3 (2026-04-13)

### Bug Fixes

- Tools result context injection.
  ([`3de8efc`](https://github.com/gadz82/orchid/commit/3de8efcd628014947d879fdbe48458ad78ec9baf))


## v1.2.2 (2026-04-13)

### Bug Fixes

- Coversation context optimization.
  ([`5dfbe5f`](https://github.com/gadz82/orchid/commit/5dfbe5f2f13c86cf39090d926c8b904c6b00dc29))


## v1.2.1 (2026-04-13)

### Bug Fixes

- Removing external dependencies and improving error handling and final outcome for the user.
  ([`8408fe2`](https://github.com/gadz82/orchid/commit/8408fe2bed57698650da3a5a4d68897250447277))


## v1.2.0 (2026-04-13)

### Features

- Orchid version bump
  ([`66dea75`](https://github.com/gadz82/orchid/commit/66dea75623a1898b761b81b1b870ddc28b3ff2a4))


## v1.0.0 (2026-04-13)

- Initial Release

## v1.1.1 (2026-04-10)

### Bug Fixes

- Remove useless Dockerfiles for orchid and orchid-api
  ([`19476b4`](https://github.com/gadz82/orchid/commit/19476b4882ce36ef231353da0c570340dfb4f6b7))


## v1.1.0 (2026-04-10)

### Bug Fixes

- Update import paths and ensure version consistency after package rename to `orchid_ai`
  ([`f6838fe`](https://github.com/gadz82/orchid/commit/f6838fe87a7757950252b30f90fdc819d904d4c6))

- **core**: Rename package root dir.
  ([`c115975`](https://github.com/gadz82/orchid/commit/c11597518f393b16973f5775ddb7d4fd319a7a28))

### Features

- Update package name from `orchid` to `orchid-ai` and adjust versioning consistency
  ([`1156c4a`](https://github.com/gadz82/orchid/commit/1156c4a28334e265592cd217cb585fcbc7d9f7d9))


## v1.0.0 (2026-04-10)

- Initial Release
