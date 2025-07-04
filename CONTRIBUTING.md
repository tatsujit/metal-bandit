# Contributing to MetalBandit

Thank you for your interest in contributing to MetalBandit! This project focuses on defensive security research and educational applications.

## Code of Conduct

This project is dedicated to providing a harassment-free experience for everyone. We are committed to creating a welcoming environment for all contributors.

## Security and Ethics Policy

**IMPORTANT**: This project is intended for:
- ✅ Defensive security research
- ✅ Educational and academic applications  
- ✅ Performance optimization research
- ✅ Statistical computing method development

**NOT intended for**:
- ❌ Malicious applications
- ❌ Unauthorized system access
- ❌ Any potentially harmful use cases

All contributions must maintain this defensive focus.

## How to Contribute

### Reporting Issues

1. Check existing issues first
2. Use the issue template
3. Provide minimal reproducible examples
4. Include system information (Julia version, macOS version, hardware)

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Write tests** for your changes
4. **Ensure all tests pass** (`julia test/runtests.jl`)
5. **Follow coding standards** (see below)
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Coding Standards

#### Julia Style Guidelines

- Use 4 spaces for indentation
- Line length: 92 characters maximum
- Use descriptive variable names
- Add docstrings for public functions
- Follow Julia naming conventions:
  - Functions: `snake_case`
  - Types: `PascalCase`
  - Constants: `UPPER_CASE`

#### Performance Guidelines

- Use `Float32` for GPU computations
- Minimize GPU-CPU data transfers
- Profile performance-critical code
- Add benchmarks for new features

#### GPU Programming Guidelines

- Use appropriate thread block sizes
- Consider memory access patterns
- Handle GPU memory efficiently
- Provide CPU fallbacks where possible

### Testing Requirements

All contributions must:

1. **Pass existing tests**: `julia test/runtests.jl`
2. **Include new tests** for new functionality
3. **Maintain test coverage** for modified code
4. **Test on Apple Silicon** (required for GPU features)

### Documentation Requirements

- Update README.md for new features
- Add docstrings for public APIs
- Update manual.org for user-facing changes
- Include examples for complex features

## Development Setup

### Requirements

- Julia 1.9+
- Apple Silicon Mac (for GPU features)
- Git
- GitHub CLI (optional, for easier PR management)

### Local Development

```bash
# Clone your fork
git clone https://github.com/yourusername/metal-bandit.git
cd metal-bandit

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia test/runtests.jl

# Run a quick demo
julia -e 'include("metal_bandit_simulator.jl"); demonstrate_metal_bandit_simulator()'
```

### Testing Locally

```bash
# Full test suite
julia test/runtests.jl

# Individual test components
julia test/test_environment.jl
julia test/test_agent.jl
julia test/test_kernels.jl
julia test/test_integration.jl
julia test/test_performance.jl
```

## Types of Contributions

### Welcome Contributions

- **Performance optimizations**: GPU kernel improvements, memory efficiency
- **Algorithm enhancements**: New bandit algorithms, better parameter estimation
- **Documentation improvements**: Clearer examples, tutorials, API docs
- **Test coverage**: Additional test cases, edge case coverage
- **Platform support**: Better cross-platform compatibility
- **Visualization enhancements**: Better plots, interactive features
- **Bug fixes**: Addressing issues and edge cases

### Ideas for Major Features

- **Additional bandit algorithms**: Thompson sampling, UCB variants
- **Extended problem types**: Gaussian bandits, contextual bandits
- **Advanced analysis tools**: Bayesian analysis, uncertainty quantification
- **Streaming computation**: Online learning capabilities
- **Multi-GPU support**: Scaling to larger problems

## Pull Request Process

1. **Ensure CI passes**: All automated tests must pass
2. **Code review**: At least one maintainer review required
3. **Documentation**: Update relevant documentation
4. **Performance**: No significant performance regressions
5. **Security review**: Ensure defensive focus is maintained

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Performance impact assessed
- [ ] Security implications considered
- [ ] Defensive use case maintained

## Release Process

1. **Version bump**: Update version in Project.toml
2. **Changelog**: Update with new features and fixes
3. **Tag release**: Create Git tag
4. **GitHub release**: Create release notes
5. **Performance benchmarks**: Update performance documentation

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check manual.org and README.md first

## Attribution

Contributors will be recognized in:
- AUTHORS file
- Release notes
- GitHub contributors page

Thank you for helping make MetalBandit better while maintaining its focus on defensive security and educational applications!