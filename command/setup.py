from setuptools import setup, find_packages

setup(
    name='classify-news',  # プロジェクト名
    version='0.1',  # バージョン
    packages=find_packages(),  # パッケージを自動的に見つけてインストールします
    install_requires=[  # 依存関係のリスト
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'datasets',
        'pyyaml'
    ],
    entry_points={  # コマンドラインツールとして使いたいスクリプトを指定します
        'console_scripts': [
            'classify-news=train:main',  # 'your_project_name'というコマンドでtrain.pyのmain関数が実行されます
        ],
    },
)
