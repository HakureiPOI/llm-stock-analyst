# 项目设置指南

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 升级 pip
pip install --upgrade pip
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key 和 Token
# - DASHSCOPE_API_KEY: 从 https://dashscope.console.aliyun.com/apiKey 获取
# - TUSHARE_TOKEN: 从 https://tushare.pro/user/token 获取
```

### 4. 验证安装

```bash
# 运行单元测试
pytest tests/ -v

# 检查依赖
python -c "from utils.validators import validate_stock_code; print(validate_stock_code('600519.SH'))"
```

### 5. 启动服务

```bash
# 开发模式
langgraph dev

# 或者使用 API 模式
# langgraph up
```

---

## 项目结构

```
llm-stock-analyst/
├── agents/                 # 智能体模块
│   ├── base.py            # 基础上下文和配置
│   ├── supervisor_entry.py # 主智能体入口
│   ├── stock_specialist.py # 股票专家
│   ├── index_specialist.py # 指数专家
│   ├── volatility_specialist.py # 波动率专家
│   └── recommendation_specialist.py # 推荐专家
├── tools/                  # LangChain 工具
│   ├── stock_tools.py     # 股票相关工具
│   ├── index_tools.py     # 指数相关工具
│   ├── analysis_tools.py  # 技术分析工具
│   ├── volatility_tools.py # 波动率预测工具
│   ├── recommendation_tools.py # 推荐工具
│   └── websearch_tools.py # 联网搜索工具
├── tsdata/                 # 数据源层
│   ├── stock.py           # 股票数据接口
│   ├── index.py           # 指数数据接口
│   ├── indicators.py      # 技术指标计算
│   └── cache.py           # 数据缓存
├── ml/                     # 波动率预测模块
│   ├── predict.py         # 模型预测
│   ├── feature_engineering.py # 特征工程
│   └── ...
├── config/                 # 配置模块
│   └── models.py          # 模型配置
├── utils/                  # 工具模块（新增）
│   ├── validators.py      # 输入验证
│   ├── logger.py          # 日志配置
│   └── retry.py           # 重试和限流
├── tests/                  # 单元测试（新增）
│   ├── test_validators.py
│   └── test_cache.py
└── requirements.txt        # 依赖列表
```

---

## 主要改进

### 安全性提升

1. **API Key 验证**: 启动时验证必要的环境变量
2. **输入验证**: 所有股票代码、日期等输入都经过严格验证
3. **错误处理**: 完善的异常捕获和日志记录

### 代码质量

1. **类型提示**: 完整的类型注解
2. **单元测试**: 覆盖核心功能
3. **日志系统**: 统一的日志管理
4. **缓存优化**: 修复缓存键碰撞问题

### 稳定性

1. **连接管理**: 正确管理 PostgreSQL 连接生命周期
2. **重试机制**: API 调用自动重试
3. **限流保护**: 防止 API 超限

---

## 常见问题

### Q: 启动时提示 "DASHSCOPE_API_KEY 环境变量未设置"

A: 确保 `.env` 文件存在且包含有效的 `DASHSCOPE_API_KEY`：
```bash
# 检查 .env 文件
cat .env | grep DASHSCOPE_API_KEY

# 如果没有，复制示例配置并编辑
cp .env.example .env
# 然后编辑 .env 文件
```

### Q: 单元测试失败

A: 首先确保安装了测试依赖：
```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Q: 如何查看日志

A: 日志默认输出到控制台。如需文件日志，在 `.env` 中配置：
```bash
LOG_FILE=logs/stock_analyst.log
```

### Q: PostgreSQL 连接失败

A: 如果不使用持久化存储，可以忽略。如需使用，配置 `DB_URI`：
```bash
DB_URI=postgresql://user:password@localhost:5432/stock_analyst
```

---

## 开发指南

### 添加新的工具

1. 在 `tools/` 目录创建新文件
2. 使用 `@tool` 装饰器定义工具
3. 添加工具到 `tools/__init__.py`
4. 编写单元测试

### 添加新的验证规则

1. 在 `utils/validators.py` 添加验证函数
2. 返回 `(bool, str)` 元组：`(是否有效，错误信息)`
3. 在工具中调用验证函数

### 运行特定测试

```bash
# 运行单个测试文件
pytest tests/test_validators.py -v

# 运行特定测试类
pytest tests/test_validators.py::TestValidateStockCode -v

# 运行特定测试函数
pytest tests/test_validators.py::TestValidateStockCode::test_valid_sh_stock -v
```

---

## 性能优化建议

1. **缓存**: 数据查询自动缓存 1 小时
2. **批量查询**: 支持逗号分隔的多个股票代码
3. **限制返回**: 使用 `limit` 参数限制返回记录数
4. **日期范围**: 指定合理的日期范围，避免全量查询

---

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

MIT License

---

## 联系方式

如有问题，请提交 Issue 或联系维护者。
