# ComfyUI-ImageAtlas

将多张图片打包成纹理地图集（Texture Atlas）的 ComfyUI 节点。

## 功能特性

- **智能透明像素分析**：自动检测并裁剪图片中的透明边缘
- **多种打包算法**：支持 7 种知名矩形打包算法
- **可配置间隙**：支持图片间留白设置，默认1像素
- **元数据存储**：支持将位置信息存储在PNG自定义块中
  - `aTLS`：明文格式，逗号分隔
  - `aTLZ`：二进制格式，每个int 2字节（范围0~65535）

## 支持的打包算法

| 算法 | 说明 | 特点 |
|------|------|------|
| **Shelf (架子算法)** | 像放书架一样一行行放置 | 简单快速，适合相似尺寸图片 |
| **MaxRects-BSSF** | 最短边适配 | 优先匹配短边，紧凑度高 |
| **MaxRects-BLSF** | 最长边适配 | 优先匹配长边 |
| **MaxRects-BAF** | 最佳面积适配 | 优先填满空间，利用率高 |
| **MaxRects-BL** | 左下角优先 | 优先放置在左下角 |
| **Skyline (天际线)** | 维护天际线轮廓 | 比Shelf更紧凑 |
| **Guillotine (切割)** | 横切/竖切分割空间 | 适合规则图片 |

## 节点说明

### 🗺️ 纹理地图集生成 (ImageAtlas)

将多张输入图片打包成一张纹理地图集。

**输入：**
- `images`：输入图片批次
- `max_width`：地图集最大宽度（默认2048）
- `gap`：图片间隙像素（默认1）
- `algorithm`：打包算法（7种可选）
- `metadata_format`：元数据格式（aTLS明文/aTLZ二进制）
- `trim_transparent`：是否裁剪透明边缘（默认开启）

**输出：**
- `atlas_image`：合并后的纹理图
- `metadata_json`：JSON格式的元数据

### 💾 保存纹理地图集 (ImageAtlasSave)

保存纹理地图集为带有自定义PNG块的文件。

### ✂️ 从地图集提取图片 (ImageAtlasExtract)

根据索引从地图集中提取指定图片。

### 📂 加载纹理地图集 (ImageAtlasLoad)

从PNG文件读取地图集及其元数据。

### 📦 地图集还原为多图 (ImageAtlasLoader)

从带有 aTLS/aTLZ 块的PNG中提取所有原始图片。

**输入：**
- `image_path`：PNG文件路径
- `original_width`：原始图片宽度
- `original_height`：原始图片高度

**输出：**
- `images`：还原后的图片批次（每张图片恢复到原始尺寸和位置）
- `count`：图片数量

**注意：** 如果PNG文件中没有 aTLS/aTLZ 块，将报错。

## 元数据格式

每张图片存储6个值：
- `orig_x`, `orig_y`, `orig_w`, `orig_h`：原始图片中非透明区域的位置和尺寸
- `atlas_x`, `atlas_y`：在合并纹理中的位置

### aTLS 格式（明文）

```
x1,y1,w1,h1,nx1,ny1,x2,y2,w2,h2,nx2,ny2,...
```

### aTLZ 格式（二进制）

每个值占2字节（big-endian unsigned short），坐标范围0~65535。

## 安装

1. 将 `ComfyUI-ImageAtlas` 文件夹复制到 ComfyUI 的 `custom_nodes` 目录
2. 重启 ComfyUI

## 算法说明

### Shelf (架子算法)
像放书架一样一行行放置，按高度排序后从左到右填充。

### MaxRects (最大矩形算法)
维护空闲矩形列表，每次选择最佳位置放置：
- **BSSF**：优先匹配短边
- **BLSF**：优先匹配长边
- **BAF**：优先填满面积
- **BL**：优先左下角

### Skyline (天际线算法)
维护一条天际线轮廓，找最低的坑位放置，比 Shelf 更紧凑。

### Guillotine (切割算法)
每次放置后将剩余空间横切或竖切成两个矩形。

## 作者

blueshell

