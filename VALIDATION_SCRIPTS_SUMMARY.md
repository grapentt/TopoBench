## Your Role

You are continuing work on the TopoBench Challenge 2025 submissions for **B1 (Inductive)**

## Before Starting ANY Work

1. **Read SHORTTERM.md** to see:
   - What were the last 2 tasks completed
   - What is the next task to work on

2. **Read GOAL.md** to understand:
   - Overall project status
   - What needs to be done
   - Architecture decisions

3. **Briefly review LONGTERM.md** for:
   - Strategic vision
   - Design principles
   - Any architectural constraints

## Your Workflow

### 1. Plan
- Understand the next task from SHORTTERM.md
- If unclear or blocked, ask for clarification
- If SHORTTERM.md is empty or outdated, check GOAL.md for priorities

### 2. Implement
- Write clean, typed, documented code (PEP8 compliant)
- Use `.venv/bin/python` for all Python commands
- NO git operations
- Test your code as you go
- Follow TopoBench patterns and conventions

### 3. Test
- Run relevant tests
- Verify functionality
- Check for errors/warnings
- Ensure integration with TopoBench works

### 4. Document
**ALWAYS update these documents after completing work:**

#### **SHORTTERM.md** (REQUIRED after each task)
- Move current "next task" to "completed tasks"
- Add new "next task" based on what makes sense
- Only keep last 2 completed + next 1 to do
- Add any blockers or important notes

#### **LONGTERM.md** (Update when relevant)
- Add architectural insights if you learned something important
- Update progress tracking if you hit a major milestone
- Add lessons learned if you encountered significant issues
- Don't inflate - only add valuable information

#### **GOAL.md** (Update when status changes)
- Mark items as complete
- Add new gaps/issues if discovered
- Update recommendations if architecture changes
- Keep this accurate and current

#### **GUIDE.md** (Update when user-facing changes occur)
- Add new features to usage guide
- Update examples if APIs change
- Add troubleshooting tips if you discover common issues

#### **PR_COMMIT.md** (Update continuously)
- Add files that should be committed (with clear commit messages)
- Update PR descriptions for B1 and B1 Bonus sections
- Keep track of what's ready for submission
- Remove files if you refactor/delete them

## Critical Rules

### Code Quality
- **Type hints** - Add type annotations to all functions
- **Docstrings** - Document all classes and public methods
- **PEP8** - Follow Python style guidelines
- **DRY** - Don't repeat yourself
- **KISS** - Keep it simple, stupid

### Testing
- **Test everything** - Don't assume code works
- **Use venv** - Always `.venv/bin/python`
- **Show output** - Share relevant test results
- **Fix bugs immediately** - Don't leave broken code

### Documentation
- **Update after work** - Not before, not later
- **Be accurate** - Remove outdated information
- **Be concise** - No fluff, just valuable info
- **Be honest** - Document issues/blockers

### Integration
- **Follow TopoBench patterns** - Study existing code
- **Don't break things** - Ensure backward compatibility when possible
- **Elegant extensions** - Make it feel like it was always part of TopoBench
- **Test integration** - Verify your code works with TopoBench workflow

## Common Patterns in TopoBench

### Dataset Classes
- Inherit from `InMemoryDataset` or `OnDiskDataset` (torch_geometric)
- Implement: `raw_file_names`, `processed_file_names`, `download()`, `process()`
- Store in `topobench/data/datasets/`

### Loader Classes
- Inherit from `AbstractLoader` (from `topobench/data/loaders/base.py`)
- Implement: `load_dataset()`, return (dataset, data_dir) tuple
- Store in `topobench/data/loaders/`

### Preprocessor Classes
- See `PreProcessor` in `topobench/data/preprocessor/preprocessor.py`
- Handle lifting (adding higher-order structures)
- Work with TBDataloader for training

## Workflow Example

```python
# User's perspective (similar to tutorial_model.ipynb)
from topobench.data.loaders import MyOnDiskLoader
from topobench.data.preprocessor import OnDiskPreProcessor  # or similar
from topobench.dataloader import TBDataloader
from topobench.model import TBModel

# 1. Load dataset
loader = MyOnDiskLoader(config)
dataset, data_dir = loader.load()

# 2. Preprocess (lift to higher-order structures) 
preprocessor = OnDiskPreProcessor(dataset, config)  # Should work on-disk!
lifted_dataset = preprocessor.transform(dataset)

# 3. Create dataloader
dataloader = TBDataloader(lifted_dataset, config)

# 4. Train model
model = TBModel(config)
trainer.fit(model, dataloader)
```

## What to Do If...

### You're unsure what to work on next
- Check SHORTTERM.md for next task
- If empty, check GOAL.md for priorities
- If still unclear, ask for guidance

### You discover a bug
- Fix it immediately
- Test the fix
- Update SHORTTERM.md to note the bug fix
- Update PR_COMMIT.md with files changed

### You need to refactor
- Document the reason in LONGTERM.md
- Update affected files
- Update all 5 documents to remove outdated references
- Test thoroughly

### You complete a major milestone
- Update LONGTERM.md with progress
- Update GOAL.md to mark items complete
- Update SHORTTERM.md with next steps
- Celebrate (briefly) then continue

### You find a better approach
- Document why in LONGTERM.md
- Refactor if it's worth it
- Update all documents
- Don't leave half-finished refactors

## Remember

- **Quality over speed** - Get it right, not fast
- **Document as you go** - Don't leave it for later
- **Test everything** - Assumptions break things
- **Think before coding** - Plan, then implement
- **Integrate elegantly** - Make TopoBench better, not messier

## Your Task Now

1. Read SHORTTERM.md to see what's next
2. Implement that task following the guidelines above
3. Test thoroughly
4. Update all relevant documents
5. Report what you did and what's next

Let's build something exceptional!
