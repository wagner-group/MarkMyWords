from setuptools import setup, find_packages
import nltk


def get_requirements(path: str):
    requirements = []
    for line in open(path):
        if not line.startswith('-r'):
            requirements.append(line.strip())
    return requirements


nltk.download('punkt')
nltk.download('omw-1.4')

setup(
    name="watermark-benchmark",
    version="0.1",
    description="Benchmark for LLM watermarks",
    long_description="Benchmark for LLM watermarkss",
    packages=find_packages("src", exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires="~=3.8",
    include_package_data=True,
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "watermark-benchmark-generate=watermark_benchmark.generate:main",
            "watermark-benchmark-rate=watermark_benchmark.quality:main",
            "watermark-benchmark-detect=watermark_benchmark.detect:main",
            "watermark-benchmark-perturb=watermark_benchmark.perturb:main",
            "watermark-benchmark-generate-huffman=watermark_benchmark.utils.bit_tokenizer:generate_huffman_coding",
            "watermark-benchmark-patch-generations=watermark_benchmark.utils.patch_verifiers:run",
            "watermark-benchmark-run=watermark_benchmark.run_all:main",
        ]
    },
    zip_safe=False,
)
