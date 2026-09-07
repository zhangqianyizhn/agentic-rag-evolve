import unittest

from agentic_rag_evolve import __version__


class PackageTest(unittest.TestCase):
    def test_package_version(self) -> None:
        self.assertEqual(__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
