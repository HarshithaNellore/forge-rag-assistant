# Building a Minimum Viable Product

An MVP exists to answer one question as cheaply as possible: will people use and value this
enough to justify building more? It is a learning tool, not a stripped-down version of the final
product.

## Scoping the MVP
- Identify the single core workflow that delivers the main value proposition. Cut everything else.
- Prefer manual or semi-automated processes behind the scenes if they let you ship faster and the
  user experience does not visibly suffer.
- Avoid building for scale, edge cases, or hypothetical future customers. Build for the first 10
  real users only.

## Technical Shortcuts Worth Taking Early
- Use managed infrastructure (hosted databases, auth providers, payment processors) instead of
  building these systems yourself.
- Reuse existing APIs and libraries rather than writing custom logic for solved problems.
- Ship with basic logging and monitoring so you can see how people actually use the product, even
  if the interface is rough.

## What Not to Cut
- Do not skip basic security and data handling practices, even in an early MVP, if you are
  touching real user data.
- Do not skip the core value proposition itself. If the product's promise is speed, an MVP that is
  slow defeats the point of testing it.

## Measuring MVP Success
Define in advance what "this is working" looks like: a retention number, a usage frequency, or a
willingness to pay. Without a predefined bar, founders tend to interpret any usage as a good sign,
which delays hard decisions about pivoting or shutting down a bad direction.

Shipping fast matters less than shipping the right narrow slice and learning from it quickly.
