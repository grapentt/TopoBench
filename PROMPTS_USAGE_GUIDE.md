# 📚 Prompts Usage Guide

**How to use the three prompts for the next AI agent**

---

## 🎯 **OVERVIEW**

Three prompts work together in sequence to complete the validation:

1. **PROMPT_1_CONTEXT_UNDERSTANDING.md** - Read and understand (30 min)
2. **PROMPT_2_EXECUTION_VALIDATION.md** - Execute the validation (2 hours)
3. **PROMPT_3_FOLLOWUP_EXECUTION.md** - Continue/debug if needed (variable)

---

## 📋 **PROMPT 1: Context Understanding**

**File:** `PROMPT_1_CONTEXT_UNDERSTANDING.md`

**Purpose:** Ensure the agent understands the existing codebase before making changes

**Use When:**
- Starting fresh with a new agent
- Agent hasn't read the handoff documentation
- Need to verify agent understanding

**Expected Duration:** 30 minutes

**Success Check:**
Agent can answer:
- What is OnDiskInductiveDataset?
- What is OnDiskTransductiveDataset?
- What scales have been validated?
- What is the SQLite overhead issue?
- How to optimize SQLite?

**Next Step:** If successful → Proceed to Prompt 2

---

## 🚀 **PROMPT 2: Execution**

**File:** `PROMPT_2_EXECUTION_VALIDATION.md`

**Purpose:** Execute the three-phase validation plan

**Use When:**
- Agent has completed Prompt 1
- Agent understands the context
- Ready to execute validation

**Expected Duration:** 2-2.5 hours

**Phases:**
1. **Phase 1:** Optimize SQLite (30 min)
2. **Phase 2:** Fair comparison (30 min)
3. **Phase 3:** Scale testing (60 min)

**Success Check:**
- Found graph size where in-memory fails but OnDisk succeeds
- Documented with measurements
- Created `LARGE_SCALE_VALIDATION_FINAL.md`

**Next Step:** If complete → Mission accomplished!

---

## 🔄 **PROMPT 3: Follow-Up**

**File:** `PROMPT_3_FOLLOWUP_EXECUTION.md`

**Purpose:** Continue execution, debug issues, or check status

**Use When:**
- Agent needs to continue incomplete work
- Previous task was interrupted
- Debugging or refinement needed
- Checking progress

**Expected Duration:** Variable (depends on what's incomplete)

**Features:**
- Checks which phase agent is in
- Provides phase-specific instructions
- Debugging guidance
- Result documentation templates

**Success Check:**
- Phase completion checkboxes
- All deliverables created
- Evidence documented

---

## 🎯 **WORKFLOW DIAGRAM**

```
START
  ↓
┌─────────────────────────────┐
│ Give Prompt 1               │
│ (Context Understanding)     │
└─────────────────────────────┘
  ↓
  Check: Agent understands?
  ├─ NO → Clarify, give Prompt 1 again
  └─ YES ↓
┌─────────────────────────────┐
│ Give Prompt 2               │
│ (Execution)                 │
└─────────────────────────────┘
  ↓
  Agent executes Phases 1-3
  ↓
  ├─ Complete? → DONE ✅
  │
  ├─ Incomplete/Stuck?
  │   ↓
  │ ┌─────────────────────────────┐
  │ │ Give Prompt 3               │
  │ │ (Follow-up)                 │
  │ └─────────────────────────────┘
  │   ↓
  │   Agent continues from where stuck
  │   ↓
  └─→ Complete? → DONE ✅
```

---

## 💡 **USAGE EXAMPLES**

### **Example 1: Fresh Start**

**You:**
"Here's your task. Read PROMPT_1_CONTEXT_UNDERSTANDING.md"

**Agent:** Reads, understands, confirms understanding

**You:**
"Good. Now execute PROMPT_2_EXECUTION_VALIDATION.md"

**Agent:** Executes all three phases, creates final report

**Result:** ✅ Complete!

---

### **Example 2: Agent Gets Stuck**

**You:**
"Here's your task. Read PROMPT_1_CONTEXT_UNDERSTANDING.md"

**Agent:** Reads, understands

**You:**
"Execute PROMPT_2_EXECUTION_VALIDATION.md"

**Agent:** Completes Phase 1, starts Phase 2, gets error

**You:**
"Read PROMPT_3_FOLLOWUP_EXECUTION.md and continue"

**Agent:** 
- Checks status (Phase 2 incomplete)
- Debugs the error
- Completes Phase 2
- Continues to Phase 3

**Result:** ✅ Complete!

---

### **Example 3: Resuming After Interruption**

**You:**
"Read PROMPT_3_FOLLOWUP_EXECUTION.md and determine your status"

**Agent:**
- Checks Phase 1: ✅ Complete
- Checks Phase 2: ✅ Complete  
- Checks Phase 3: ❌ Only tested 10K and 20K

**Agent:** Continues Phase 3, tests 30K and 40K

**Result:** ✅ Complete!

---

## ⚠️ **COMMON ISSUES**

### Issue 1: Agent Skips Context
**Symptom:** Agent starts coding without understanding
**Fix:** Force Prompt 1, require understanding demonstration

### Issue 2: Agent Breaks Tests
**Symptom:** Modifications break existing tests
**Fix:** Prompt 3 has debugging guidance, run tests after each change

### Issue 3: Agent Can't Find Breaking Point
**Symptom:** In-memory doesn't fail at expected scales
**Fix:** Prompt 3 has guidance on progressive testing, try larger scales

### Issue 4: SQLite Still High Memory
**Symptom:** Optimization doesn't reduce memory enough
**Fix:** Prompt 3 has specific PRAGMA statements and debugging

---

## 📊 **WHAT EACH PROMPT PROVIDES**

### Prompt 1: Context Understanding
- ✅ Existing implementations explained
- ✅ What's been validated (3 tiers)
- ✅ What's incomplete (large-scale)
- ✅ Technical challenges (SQLite overhead)
- ✅ Understanding verification

### Prompt 2: Execution
- ✅ Three-phase plan (optimize, compare, scale)
- ✅ Specific code changes needed
- ✅ Expected results at each scale
- ✅ Success criteria clear
- ✅ Time estimates realistic

### Prompt 3: Follow-Up
- ✅ Status checking mechanism
- ✅ Phase-specific instructions
- ✅ Debugging guidance
- ✅ Result documentation templates
- ✅ Scientific integrity reminders

---

## 🎯 **KEY IMPROVEMENTS FROM ORIGINAL PROMPT**

### Original Issues:
- ❌ Mentioned "RocksDB" (we use SQLite)
- ❌ Called loaders "B1" and "B1 Bonus" (incorrect names)
- ❌ Referenced non-existent "VALIDATION_PLAN.md"
- ❌ Vague instructions
- ❌ No status checking
- ❌ No debugging guidance

### Improvements:
- ✅ Accurate technical details (SQLite, correct names)
- ✅ References actual files (HANDOFF_TO_NEXT_AGENT.md)
- ✅ Clear three-phase structure
- ✅ Specific code snippets
- ✅ Status checking mechanism
- ✅ Debugging guidance
- ✅ Scientific rigor maintained
- ✅ Realistic time estimates

---

## 🎊 **EXPECTED OUTCOME**

**After all three prompts:**
- ✅ SQLite optimized (memory reduced)
- ✅ Fair comparison implemented
- ✅ Breaking point found (30K-50K nodes)
- ✅ Evidence documented
- ✅ `LARGE_SCALE_VALIDATION_FINAL.md` created
- ✅ **Proof complete:** OnDisk works where in-memory fails

**Time to completion:** ~2.5 hours total

**Quality:** Professional, reproducible, scientifically rigorous

---

## 📞 **FOR YOU (THE USER)**

**How to use these prompts:**

1. **Start with Prompt 1:** Give it to a new agent
2. **Check understanding:** Ask agent to summarize
3. **Move to Prompt 2:** Once confirmed understanding
4. **Use Prompt 3 as needed:** For continuation, debugging, checking

**You don't need to give all three at once!**
- Prompt 1 → Check → Prompt 2 → Check → Done
- Or use Prompt 3 for any follow-up

**The prompts are self-contained:**
- Each references the necessary files
- Each has clear success criteria
- Each has debugging guidance

---

**You now have a complete, improved prompt system for the next agent!** 🚀
