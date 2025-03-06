Use this template:
```
<commit_type>(<domain_type>): <commit_message>
```

Where:
- **commit_type** could be one of the following:
    - `fix`: A bug fix
    - `chore`: Routine task or maintenance
    - `feat`: A new feature
    - `perf`: Performance improvement
    - `test`: Adding or updating tests
    - `docs`: Documentation updates
    - `refactor`: Code changes that neither fix a bug nor add a feature
    - `style`: Code style updates (formatting, missing semi-colons, etc.)
    - `build`: Changes that affect the build system or external dependencies
    - `ci`: Changes to our CI configuration files and scripts

- **domain_type** could be one of the following:
    - `numba`: 
    - `bfs`
    - `cuda`
    - `etc`

- **commit_message** should be a concise description of the changes made. It should be written in the imperative mood, e.g., "Add new parking detection algorithm".

Example:
```
feat(numba): Add two more kernels
```
