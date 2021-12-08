# Automatic creation of components

To create components using the automatic approach you need to create a `yml` file containing components definitions that can be written as:

- *Writing only the name of the function.*
    
    To use this approach the function code must be typed.

    E.g.:
    ```yml
    torch.select:
    torch.stack:
    ...
    ```

- *Writing the input args for the function.*

    This is useful in the next cases:
    - the function signature cannot be obtained
    - when we want to force a typing to the inputs arguments
        - note that the typings must be compatible with the original ones.
    - when not all the inputs should be exposed (in this case the hidden args must have by default values)
    - we want to force some args to default values

    E.g.:
    ```yml
    torch.select:
        params: {input: torch.Tensor, dim: int, index: int}
    ...
    ```

    E.g. assigning a default value to an arg:
    ```yml
    torch.select:
        params: {input: torch.Tensor, dim: int, index: int = 0}
    ...
    ```

- *Writing the output args for the function.*

    This is useful in the next cases:
    - the function signature cannot be obtained
    - when we want to assign a name for the out pins in the component.
        - This can only be done when the number of outputs is fixed.
    - when we want to force a typing to the output arguments
        - note that the typings must be compatible with the original ones.

    E.g. defining the entire signature:
    ```yml
    torch.select:
        returns: {out: torch.Tensor}
    ...
    ```

    E.g. defining the name of the output pin:
    ```yml
    torch.select:
        returns: [output_tensor]
    
    ...
    ```

    E.g. define the typing of the output:
    ```yml
    torch.select:
        returns: torch.Tensor
    ...
    ```

- *Combining the 2 previous options.*

    E.g.:
    ```yml
    torch.select:
        params: {input: torch.Tensor, dim: int, index: int}
        returns: {out: torch.Tensor}
    ...
    ```

**NOTE**: sometimes to write typing expresions special caharacters are required (e.g. `[]`) in that case remember to add `""` (e.g. `"typing.Optional[int]"`).

**NOTE 2**: all the non builtin types must be written with the origin modules. E.g. `Tensor` is incorrect, it must be written as `torch.Tensor`.