"""Isolated execution of one hash-bound capacity-study input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verified_module(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "execution source differs")
    result = ModuleType(path.stem)
    result.__file__ = str(path)
    # Execute precisely the source bytes whose hash was checked, not a second
    # open that could consume a subsequently changed script.
    exec(compile(raw,str(path),"exec"),result.__dict__)
    return result


class FrozenInput:
    def __init__(self, raw, expected):
        require(digest(raw) == expected,"execution input differs")
        self.raw = raw

    def read_bytes(self):
        return self.raw


def execute(repo, root, name, engine, prepared_sha256):
    require(sys.flags.isolated,"worker requires isolated Python (-I)")
    repo,root = repo.resolve(),root.resolve()
    prepared = (root/"prepared.json").read_bytes()
    require(digest(prepared) == prepared_sha256,"preparation changed before child execution")
    manifest = json.loads(prepared)
    hashes = {r["path"]:r["sha256"] for r in manifest["sources"]}
    here = Path(__file__).resolve()
    require(digest(here.read_bytes()) == hashes[str(here.relative_to(repo))],"worker source differs")
    launch_path = here.with_name("launch.py")
    launch = verified_module(launch_path,hashes[str(launch_path.relative_to(repo))])
    launch.validated_inputs(repo,root)
    require(name in launch.names() and engine in ("native","reference"),"unknown child execution")
    entry = next(e for e in manifest["entries"] if e["name"] == name)
    directory = root/name
    frozen = FrozenInput((directory/"request.json").read_bytes(),entry["request_sha256"])
    extension_hash = None
    if engine == "native":
        import spiraltorch as st
        extension_hash = digest(Path(st._rs.__file__).read_bytes())
        require(extension_hash == manifest["native_extension_sha256"],"execution child native extension differs")
        runner = verified_module(repo/"tools/run_byte_corpus_segments.py",hashes["tools/run_byte_corpus_segments.py"])
        runner.run(frozen,directory/"native",list(launch.STOPS),study_type=st.nn.ByteCorpusStudy)
    else:
        destination = directory/"torch.json"
        require(not destination.exists(),"reference output already exists")
        protocol = verified_module(repo/"tools/byte_corpus_study.py",hashes["tools/byte_corpus_study.py"])
        protocol.reference(SimpleNamespace(request=frozen,output=destination))
    require(digest((root/"prepared.json").read_bytes()) == prepared_sha256,"preparation changed during execution")
    launch.validated_inputs(repo,root)
    launch.write(directory/(engine+"-child.json"),dict(prepared_sha256=prepared_sha256,
                 request_sha256=digest(frozen.read_bytes()),native_extension_sha256=extension_hash,
                 engine=engine,isolated=True,input_bytes_bound=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","root"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("name","engine","prepared-sha256"):
        parser.add_argument("--"+name,required=True)
    args = parser.parse_args()
    execute(args.repo,args.root,args.name,args.engine,args.prepared_sha256)


if __name__ == "__main__":
    main()
