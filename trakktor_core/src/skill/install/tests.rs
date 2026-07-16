use super::*;

#[test]
fn skill_path_follows_the_design_layout() {
    let path = skill_path(Path::new("."), Target::Claude);
    assert_eq!(
        path,
        Path::new("./.claude/skills/trakktor/SKILL.md").to_path_buf()
    );
    assert_eq!(
        skill_path(Path::new("/home/u"), Target::Agents),
        Path::new("/home/u/.agents/skills/trakktor/SKILL.md").to_path_buf()
    );
}

#[test]
fn write_creates_dirs_then_skips_without_force() {
    let dir = tempfile::tempdir().unwrap();
    let path = skill_path(dir.path(), Target::Claude);

    assert_eq!(
        write_stub(&path, "first", false).unwrap(),
        WriteOutcome::Written
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "first");

    // Existing file, no force → skipped, content untouched.
    assert_eq!(
        write_stub(&path, "second", false).unwrap(),
        WriteOutcome::Skipped
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "first");

    // force → overwritten.
    assert_eq!(
        write_stub(&path, "second", true).unwrap(),
        WriteOutcome::Written
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "second");
}

#[test]
fn global_skill_path_requires_an_existing_agent_dir() {
    let home = tempfile::tempdir().unwrap();

    // Missing `.claude` → error, no path produced.
    match global_skill_path(home.path(), Target::Claude) {
        Err(SkillError::AgentDirMissing(dir)) => {
            assert_eq!(dir, agent_dir(home.path(), Target::Claude));
        },
        other => panic!("expected AgentDirMissing, got {other:?}"),
    }

    // Once `.claude` exists, the usual layout is returned.
    fs::create_dir(agent_dir(home.path(), Target::Claude)).unwrap();
    assert_eq!(
        global_skill_path(home.path(), Target::Claude).unwrap(),
        skill_path(home.path(), Target::Claude)
    );
}
