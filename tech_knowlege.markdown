# RunnableLambda 超清晰用法（专门给 LCEL 新手）

我用最简单、最实用、能直接套用的方式讲，保证你马上会用！

---

## 一句话核心

**RunnableLambda = 把普通函数 / lambda 变成 LCEL 能用的组件**

只有它包装过的函数，才能用 `|` 拼接、才能 `.stream()` / `.invoke()`。

---

## 1. 最基础用法（必学）

### 格式
```python
RunnableLambda(你的函数)
```

### 最简单例子
```python
from langchain_core.runnables import RunnableLambda

# 1. 写一个普通小函数
def add_one(x):
    return x + 1

# 2. 用 RunnableLambda 包装
runnable = RunnableLambda(add_one)

# 3. 现在它就是 LCEL 组件，可以调用了！
print(runnable.invoke(5))  # 输出 6
```

---

## 2. 结合 lambda 简写（你项目里那种）

你项目里的：
```python
RunnableLambda(lambda x: x["input"])
```

等价于：
```python
def get_input(x):
    return x["input"]

RunnableLambda(get_input)
```

**lambda 只是简写，功能一模一样！**

---

## 3. 最常用场景：字典里取值（你项目正在用）

```python
# 输入是字典：
x = {"input": "你好", "history": []}

# 取出 input
RunnableLambda(lambda x: x["input"]).invoke(x)
# 输出：你好

# 取出 history
RunnableLambda(lambda x: x["history"]).invoke(x)
# 输出：[]
```

**这就是你代码里的作用！**

---

## 4. 为什么 LCEL 必须用它？

因为：
- 普通函数 **不能** 用 `|` 管道拼接
- 普通函数 **不能** `.stream()`
- 普通函数 **不能** 放进 LCEL 字典里

**套上 RunnableLambda，立刻变成标准 LCEL 组件！**

---

## 5. 完整实战：放进你的 RAG 链里

你这段代码：
```python
{
    "input": RunnableLambda(lambda x: x["input"]),
    "context": wrapped_retriever,
    "history": lambda x: x["history"]
}
```

**作用：**
- 接收一个大字典 `x`
- 从 `x` 里拿出 `input` 字段
- 包装成 LCEL 能用的取值工具

---

## 6. 3 种最常用写法（直接复制用）

### ① 从字典取字段（最常用）
```python
RunnableLambda(lambda x: x["input"])
RunnableLambda(lambda x: x["question"])
RunnableLambda(lambda x: x["history"])
```

### ② 自定义处理函数
```python
def clean_text(text):
    return text.strip()

chain = RunnableLambda(clean_text) | llm
```

### ③ 链式调用
```python
chain = RunnableLambda(lambda x: x+1) | RunnableLambda(lambda x: x*2)
chain.invoke(3)  # (3+1)*2 = 8
```

---

## 7. 超级总结（背会这一句就够）

> **RunnableLambda(函数) = 把普通函数变成 LCEL 管道里的一个环节 = 专门用来取值、处理数据、自定义逻辑**

你项目里的：
```python
RunnableLambda(lambda x: x["input"])
```

就是 **从输入字典里把 input 取出来** 给后面的 prompt 用。
