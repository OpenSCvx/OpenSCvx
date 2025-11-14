"""Lowering backends for converting symbolic expressions to executable code.

This package contains backend implementations that translate openscvx's symbolic
expression AST into executable code for different computational frameworks. The
lowering process is a compilation step that happens after symbolic problem
construction but before numerical optimization.

Architecture:
    All lowerers in this package follow a common visitor pattern:

    1. **Visitor Pattern**: Each backend defines visitor methods for each expression
       type, registered via the @visitor decorator
    2. **Recursive Lowering**: Visitors recursively lower child expressions and
       compose backend-specific operations
    3. **Centralized Dispatch**: The dispatch() function routes expressions to
       their registered visitor methods
    4. **Type Safety**: Each visitor is strongly typed to its backend's output format

Available Backends:
    - **jax**: Lowers to JAX functions with automatic differentiation support
    - **cvxpy**: Lowers to CVXPy expressions for disciplined convex programming

    See individual backend modules for detailed documentation and usage examples.

For Contributors:
    **Adding a New Backend**

    To add a new lowering backend:

    1. **Create a new module** in this package (e.g., `mybackend.py`)

    2. **Implement the visitor pattern**::

        from typing import Dict, Type, Callable, Any
        from openscvx.symbolic.expr import Expr

        _MYBACKEND_VISITORS: Dict[Type[Expr], Callable] = {}

        def visitor(expr_cls: Type[Expr]):
            '''Decorator to register visitor methods.'''
            def register(fn: Callable):
                _MYBACKEND_VISITORS[expr_cls] = fn
                return fn
            return register

        def dispatch(lowerer: Any, expr: Expr):
            '''Dispatch expression to registered visitor.'''
            fn = _MYBACKEND_VISITORS.get(type(expr))
            if fn is None:
                raise NotImplementedError(
                    f"{lowerer.__class__.__name__} has no visitor for {type(expr).__name__}"
                )
            return fn(lowerer, expr)

    3. **Create the lowerer class**::

        class MyBackendLowerer:
            '''Lower symbolic expressions to MyBackend format.'''

            def lower(self, expr: Expr):
                '''Main entry point for lowering.'''
                return dispatch(self, expr)

            @visitor(Constant)
            def _visit_constant(self, node: Constant):
                # Convert to backend representation
                return mybackend.constant(node.value)

            @visitor(Add)
            def _visit_add(self, node: Add):
                # Recursively lower and combine
                terms = [self.lower(t) for t in node.terms]
                return mybackend.add(*terms)

            # Implement visitors for all expression types...

    4. **Add convenience wrapper**::

        def lower_to_mybackend(expr: Expr, **kwargs):
            '''Convenience function for lowering to MyBackend.'''
            lowerer = MyBackendLowerer(**kwargs)
            return lowerer.lower(expr)

    5. **Document thoroughly** - Follow the documentation patterns in jax.py
       and cvxpy.py, including module docstring, class docstring, and method
       docstrings for all visitor methods.

    **Key Design Patterns**

    - Use private method names for visitors: `_visit_*` (not part of public API)
    - Recursively lower child expressions using `self.lower()`
    - Keep visitor methods stateless when possible
    - Handle edge cases (scalars, empty arrays, etc.)
    - Document mathematical properties and backend-specific constraints

    **Adding Support for New Expression Types**

    See the "For Contributors" sections in the existing backend modules for
    detailed guidance on adding visitor methods for new expression types.

See Also:
    - openscvx.symbolic.lower: Main lowering orchestration functions
    - openscvx.symbolic.expr: Symbolic expression AST definitions
    - openscvx.symbolic.lowerers.jax: JAX backend documentation and implementation
    - openscvx.symbolic.lowerers.cvxpy: CVXPy backend documentation and implementation
"""
