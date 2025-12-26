# -- ComfyUI-ImageAtlas/atlas_nodes.py --
"""
纹理地图集生成节点
将多张图片打包成纹理地图集，支持透明像素分析和自定义PNG块存储元数据
"""

import torch
import numpy as np
from PIL import Image
import io
import struct
import zlib
from typing import List, Tuple, Optional


class RectangleArgs:
    """矩形参数类，存储图片的位置和尺寸信息"""
    
    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0, 
                 orig_x: int = 0, orig_y: int = 0, orig_w: int = 0, orig_h: int = 0,
                 image_index: int = 0, image_data: np.ndarray = None):
        # 在合并纹理中的位置
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # 原始图片中非透明区域的位置
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.orig_w = orig_w
        self.orig_h = orig_h
        # 原始图片索引
        self.image_index = image_index
        # 裁剪后的图片数据
        self.image_data = image_data
    
    def clone(self) -> 'RectangleArgs':
        return RectangleArgs(
            self.x, self.y, self.width, self.height,
            self.orig_x, self.orig_y, self.orig_w, self.orig_h,
            self.image_index, self.image_data
        )


class FloorPlane:
    """地板扫描纹理打包算法"""
    
    def __init__(self):
        self.rect_array: List[RectangleArgs] = []
        self.height_array: List[int] = []
        self.start_x = 0
        self.is_left_align = True
        self.width_line_sum = 0
        self.mode_switch = 0
    
    def deal_rectangle_args_array(self, a_rect_array: List[RectangleArgs], 
                                   max_width: int, gap: int = 1) -> List[RectangleArgs]:
        """处理矩形数组，进行打包布局"""
        # 克隆并排序
        self.rect_array = [r.clone() for r in a_rect_array]
        self.rect_array.sort(key=lambda r: (r.height, r.width), reverse=True)
        
        # 添加间隙
        if gap:
            for rect in self.rect_array:
                rect.width += gap
                rect.height += gap
        
        # 执行地板扫描算法
        self._floorplane(max_width)
        
        # 移除间隙
        if gap:
            for rect in self.rect_array:
                rect.width -= gap
                rect.height -= gap
        
        return self.rect_array
    
    def _count_max_h(self, width: int, start: int) -> int:
        """计算指定范围内的最大高度"""
        end = start + width
        max_h = self.height_array[start]
        for i in range(start + 1, end):
            if max_h < self.height_array[i]:
                max_h = self.height_array[i]
        return max_h
    
    def _find_min_h_left(self, width: int) -> Tuple[int, int]:
        """从左到右找最小高度位置"""
        length = len(self.height_array) - width
        min_x = 0
        min_h = 99999999
        
        for i in range(length):
            h = self._count_max_h(width, i)
            if h < min_h:
                min_x = i
                min_h = h
        
        return min_x, min_h
    
    def _find_min_h_right(self, width: int) -> Tuple[int, int]:
        """从右到左找最小高度位置"""
        length = len(self.height_array) - width
        min_x = 0
        min_h = 99999999
        
        for i in range(length - 1, -1, -1):
            h = self._count_max_h(width, i)
            if h < min_h:
                min_x = i
                min_h = h
        
        return min_x, min_h
    
    def _floorplane(self, max_width: int):
        """地板扫描打包算法主体"""
        total = len(self.rect_array)
        mode1 = 1
        
        self.height_array = [0] * max_width
        self.start_x = 0
        self.is_left_align = True
        self.width_line_sum = 0
        
        import math
        self.mode_switch = int(math.log2(max_width)) if max_width > 0 else 0
        
        bi = 0
        while bi < total:
            rect = self.rect_array[bi]
            
            if self.width_line_sum + rect.width > max_width:
                rest_space = max_width - self.width_line_sum
                
                # 尝试找一个能放进剩余空间的小矩形
                for bi2 in range(bi + 1, total):
                    if self.rect_array[bi2].width <= rest_space:
                        rect2 = self.rect_array.pop(bi2)
                        self.rect_array.insert(bi, rect2)
                        break
                
                if rect != self.rect_array[bi]:
                    continue
                else:
                    self.is_left_align = not self.is_left_align
                    self.width_line_sum = 0
                    self.start_x = 0 if self.is_left_align else max_width
                    continue
            else:
                bi += 1
                self.width_line_sum += rect.width
                
                if total - bi < self.mode_switch:
                    mode1 = 0
                
                if mode1:
                    if self.is_left_align:
                        start_height = self._count_max_h(rect.width, self.start_x)
                        start_x = self.start_x
                        self.start_x += rect.width
                    else:
                        start_height = self._count_max_h(rect.width, self.start_x - rect.width)
                        start_x = self.start_x - rect.width
                        self.start_x -= rect.width
                else:
                    if self.is_left_align:
                        start_x, start_height = self._find_min_h_left(rect.width)
                    else:
                        start_x, start_height = self._find_min_h_right(rect.width)
                
                rect.x = start_x
                rect.y = start_height
                x_max = start_x + rect.width
                h_max = start_height + rect.height
                
                for i in range(start_x, x_max):
                    self.height_array[i] = h_max


class MaxRects:
    """
    MaxRects 最大矩形算法
    维护空闲矩形列表，每次选择最佳位置放置
    支持多种启发式策略：BSSF, BLSF, BAF, BL
    """
    
    def __init__(self):
        self.free_rects: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
        self.bin_width = 0
        self.bin_height = 0
    
    def deal_rectangle_args_array(self, a_rect_array: List[RectangleArgs], 
                                   max_width: int, gap: int = 1,
                                   heuristic: str = "BSSF") -> List[RectangleArgs]:
        """处理矩形数组，进行打包布局"""
        rect_array = [r.clone() for r in a_rect_array]
        
        # 添加间隙
        if gap:
            for rect in rect_array:
                rect.width += gap
                rect.height += gap
        
        # 使用动态高度，初始只有最大图片高度
        max_rect_height = max(r.height for r in rect_array) if rect_array else 1
        
        self.bin_width = max_width
        self.bin_height = max_rect_height
        self.free_rects = [(0, 0, max_width, max_rect_height)]
        
        # 按面积降序排序
        rect_array.sort(key=lambda r: r.width * r.height, reverse=True)
        
        for rect in rect_array:
            best_score = 99999999
            best_score2 = 99999999  # 次要评分：y坐标
            best_x, best_y = 0, 0
            best_idx = -1
            
            for idx, (fx, fy, fw, fh) in enumerate(self.free_rects):
                if rect.width <= fw and rect.height <= fh:
                    # 计算评分
                    if heuristic == "BSSF":  # Best Short Side Fit
                        score = min(fw - rect.width, fh - rect.height)
                        score2 = fy  # 优先选择 y 坐标较小的
                    elif heuristic == "BLSF":  # Best Long Side Fit
                        score = max(fw - rect.width, fh - rect.height)
                        score2 = fy
                    elif heuristic == "BAF":  # Best Area Fit
                        score = fw * fh - rect.width * rect.height
                        score2 = fy
                    else:  # BL - Bottom Left
                        score = fy
                        score2 = fx
                    
                    # 主评分更小，或主评分相同但次要评分更小
                    if score < best_score or (score == best_score and score2 < best_score2):
                        best_score = score
                        best_score2 = score2
                        best_x, best_y = fx, fy
                        best_idx = idx
            
            if best_idx == -1:
                # 扩展高度，在底部添加新的空闲区域
                new_y = self.bin_height
                self.bin_height += rect.height
                self.free_rects.append((0, new_y, max_width, rect.height))
                best_x, best_y = 0, new_y
                best_idx = len(self.free_rects) - 1
            
            rect.x = best_x
            rect.y = best_y
            
            # 分割空闲矩形
            self._split_free_rect(best_idx, rect.x, rect.y, rect.width, rect.height)
            self._merge_free_rects()
        
        # 移除间隙
        if gap:
            for rect in rect_array:
                rect.width -= gap
                rect.height -= gap
        
        return rect_array
    
    def _split_free_rect(self, idx: int, px: int, py: int, pw: int, ph: int):
        """分割空闲矩形"""
        new_rects = []
        i = 0
        while i < len(self.free_rects):
            fx, fy, fw, fh = self.free_rects[i]
            
            # 检查是否与放置的矩形相交
            if px < fx + fw and px + pw > fx and py < fy + fh and py + ph > fy:
                # 左边剩余
                if px > fx:
                    new_rects.append((fx, fy, px - fx, fh))
                # 右边剩余
                if px + pw < fx + fw:
                    new_rects.append((px + pw, fy, fx + fw - px - pw, fh))
                # 上边剩余
                if py > fy:
                    new_rects.append((fx, fy, fw, py - fy))
                # 下边剩余
                if py + ph < fy + fh:
                    new_rects.append((fx, py + ph, fw, fy + fh - py - ph))
                
                self.free_rects.pop(i)
            else:
                i += 1
        
        self.free_rects.extend(new_rects)
    
    def _merge_free_rects(self):
        """合并可合并的空闲矩形，移除被包含的矩形"""
        i = 0
        while i < len(self.free_rects):
            j = i + 1
            while j < len(self.free_rects):
                ri = self.free_rects[i]
                rj = self.free_rects[j]
                
                # 检查 ri 是否包含 rj
                if (ri[0] <= rj[0] and ri[1] <= rj[1] and 
                    ri[0] + ri[2] >= rj[0] + rj[2] and 
                    ri[1] + ri[3] >= rj[1] + rj[3]):
                    self.free_rects.pop(j)
                # 检查 rj 是否包含 ri
                elif (rj[0] <= ri[0] and rj[1] <= ri[1] and 
                      rj[0] + rj[2] >= ri[0] + ri[2] and 
                      rj[1] + rj[3] >= ri[1] + ri[3]):
                    self.free_rects.pop(i)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1


class Skyline:
    """
    Skyline 天际线算法
    维护一条天际线轮廓，找最低的坑位放置矩形
    """
    
    def __init__(self):
        self.skyline: List[Tuple[int, int, int]] = []  # (x, y, width)
        self.bin_width = 0
    
    def deal_rectangle_args_array(self, a_rect_array: List[RectangleArgs], 
                                   max_width: int, gap: int = 1) -> List[RectangleArgs]:
        """处理矩形数组，进行打包布局"""
        rect_array = [r.clone() for r in a_rect_array]
        
        # 添加间隙
        if gap:
            for rect in rect_array:
                rect.width += gap
                rect.height += gap
        
        self.bin_width = max_width
        self.skyline = [(0, 0, max_width)]  # 初始天际线
        
        # 按高度降序排序
        rect_array.sort(key=lambda r: (r.height, r.width), reverse=True)
        
        for rect in rect_array:
            best_x, best_y = self._find_best_position(rect.width, rect.height)
            rect.x = best_x
            rect.y = best_y
            self._add_to_skyline(best_x, best_y, rect.width, rect.height)
        
        # 移除间隙
        if gap:
            for rect in rect_array:
                rect.width -= gap
                rect.height -= gap
        
        return rect_array
    
    def _find_best_position(self, width: int, height: int) -> Tuple[int, int]:
        """找到最佳放置位置（最低且最左）"""
        best_x, best_y = 0, 99999999
        best_waste = 99999999
        
        for i, (sx, sy, sw) in enumerate(self.skyline):
            if sw >= width:
                # 检查这个位置的实际高度（考虑右边可能更高的天际线）
                y = self._get_height_at(sx, width)
                waste = y - sy  # 浪费的空间
                
                if y < best_y or (y == best_y and waste < best_waste):
                    best_x = sx
                    best_y = y
                    best_waste = waste
            
            # 尝试跨越多个天际线段
            if i < len(self.skyline) - 1:
                combined_width = sw
                max_y = sy
                for j in range(i + 1, len(self.skyline)):
                    combined_width += self.skyline[j][2]
                    max_y = max(max_y, self.skyline[j][1])
                    if combined_width >= width:
                        if max_y < best_y:
                            best_x = sx
                            best_y = max_y
                        break
        
        return best_x, best_y
    
    def _get_height_at(self, x: int, width: int) -> int:
        """获取指定位置和宽度范围内的最大高度"""
        max_y = 0
        for sx, sy, sw in self.skyline:
            if sx < x + width and sx + sw > x:
                max_y = max(max_y, sy)
        return max_y
    
    def _add_to_skyline(self, x: int, y: int, width: int, height: int):
        """将矩形添加到天际线"""
        new_y = y + height
        new_skyline = []
        
        for sx, sy, sw in self.skyline:
            # 完全在左边
            if sx + sw <= x:
                new_skyline.append((sx, sy, sw))
            # 完全在右边
            elif sx >= x + width:
                new_skyline.append((sx, sy, sw))
            else:
                # 有重叠，需要分割
                # 左边部分
                if sx < x:
                    new_skyline.append((sx, sy, x - sx))
                # 右边部分
                if sx + sw > x + width:
                    new_skyline.append((x + width, sy, sx + sw - x - width))
        
        # 添加新的天际线段
        new_skyline.append((x, new_y, width))
        
        # 按x排序
        new_skyline.sort(key=lambda s: s[0])
        
        # 合并相邻的相同高度段
        merged = []
        for seg in new_skyline:
            if merged and merged[-1][1] == seg[1] and merged[-1][0] + merged[-1][2] == seg[0]:
                # 合并
                merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + seg[2])
            else:
                merged.append(seg)
        
        self.skyline = merged


class Guillotine:
    """
    Guillotine 切割算法
    每次放置后将剩余空间横切或竖切成两个矩形
    """
    
    def __init__(self):
        self.free_rects: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
    
    def deal_rectangle_args_array(self, a_rect_array: List[RectangleArgs], 
                                   max_width: int, gap: int = 1,
                                   split_method: str = "shorter_axis") -> List[RectangleArgs]:
        """处理矩形数组，进行打包布局"""
        rect_array = [r.clone() for r in a_rect_array]
        
        # 添加间隙
        if gap:
            for rect in rect_array:
                rect.width += gap
                rect.height += gap
        
        # 估算初始高度
        total_area = sum(r.width * r.height for r in rect_array)
        estimated_height = max(int(total_area / max_width * 1.5), max(r.height for r in rect_array))
        
        self.free_rects = [(0, 0, max_width, estimated_height)]
        
        # 按面积降序排序
        rect_array.sort(key=lambda r: r.width * r.height, reverse=True)
        
        for rect in rect_array:
            best_idx = -1
            best_score = 99999999
            
            for idx, (fx, fy, fw, fh) in enumerate(self.free_rects):
                if rect.width <= fw and rect.height <= fh:
                    # Best Area Fit
                    score = fw * fh
                    if score < best_score:
                        best_score = score
                        best_idx = idx
            
            if best_idx == -1:
                # 扩展高度
                new_height = rect.height
                self.free_rects.append((0, estimated_height, max_width, new_height))
                estimated_height += new_height
                best_idx = len(self.free_rects) - 1
            
            fx, fy, fw, fh = self.free_rects[best_idx]
            rect.x = fx
            rect.y = fy
            
            # 移除使用的空闲矩形
            self.free_rects.pop(best_idx)
            
            # 分割剩余空间
            remaining_w = fw - rect.width
            remaining_h = fh - rect.height
            
            if remaining_w > 0 or remaining_h > 0:
                if split_method == "shorter_axis":
                    # 沿较短轴分割
                    if remaining_w < remaining_h:
                        # 水平分割
                        if remaining_w > 0:
                            self.free_rects.append((fx + rect.width, fy, remaining_w, rect.height))
                        if remaining_h > 0:
                            self.free_rects.append((fx, fy + rect.height, fw, remaining_h))
                    else:
                        # 垂直分割
                        if remaining_h > 0:
                            self.free_rects.append((fx, fy + rect.height, rect.width, remaining_h))
                        if remaining_w > 0:
                            self.free_rects.append((fx + rect.width, fy, remaining_w, fh))
                else:
                    # 沿较长轴分割
                    if remaining_w > remaining_h:
                        if remaining_w > 0:
                            self.free_rects.append((fx + rect.width, fy, remaining_w, rect.height))
                        if remaining_h > 0:
                            self.free_rects.append((fx, fy + rect.height, fw, remaining_h))
                    else:
                        if remaining_h > 0:
                            self.free_rects.append((fx, fy + rect.height, rect.width, remaining_h))
                        if remaining_w > 0:
                            self.free_rects.append((fx + rect.width, fy, remaining_w, fh))
        
        # 移除间隙
        if gap:
            for rect in rect_array:
                rect.width -= gap
                rect.height -= gap
        
        return rect_array


def find_opaque_bounds(image: np.ndarray, trim_mode: str = "auto", 
                       white_threshold: int = 250) -> Tuple[int, int, int, int]:
    """
    找出图片中非透明/非白色区域的边界
    返回 (x, y, width, height)
    
    Args:
        image: 输入图片 [H, W, C]
        trim_mode: 裁剪模式
            - "auto": 有alpha用alpha，否则用白色检测
            - "alpha": 仅按alpha通道裁剪
            - "white": 按白色像素裁剪
            - "both": 同时考虑alpha和白色
        white_threshold: 白色阈值（RGB都大于此值视为白色）
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    has_alpha = channels >= 4
    
    # 根据模式决定检测方式
    if trim_mode == "alpha" and has_alpha:
        # 仅按alpha
        mask = image[:, :, 3] > 0
    elif trim_mode == "white":
        # 仅按白色
        if channels >= 3:
            mask = ~((image[:, :, 0] >= white_threshold) & 
                     (image[:, :, 1] >= white_threshold) & 
                     (image[:, :, 2] >= white_threshold))
        else:
            mask = image[:, :, 0] < white_threshold
    elif trim_mode == "both" and has_alpha:
        # 同时考虑alpha和白色
        alpha_mask = image[:, :, 3] > 0
        white_mask = ~((image[:, :, 0] >= white_threshold) & 
                       (image[:, :, 1] >= white_threshold) & 
                       (image[:, :, 2] >= white_threshold))
        mask = alpha_mask & white_mask
    else:  # auto
        if has_alpha:
            # 有alpha，用alpha
            mask = image[:, :, 3] > 0
        else:
            # 无alpha，检测白色边缘
            if channels >= 3:
                mask = ~((image[:, :, 0] >= white_threshold) & 
                         (image[:, :, 1] >= white_threshold) & 
                         (image[:, :, 2] >= white_threshold))
            else:
                mask = image[:, :, 0] < white_threshold
    
    # 找非空像素
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # 完全空白，返回空区域
        return 0, 0, 0, 0
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def create_png_with_custom_chunk(image: Image.Image, chunk_type: str, chunk_data: bytes) -> bytes:
    """
    创建带有自定义块的PNG文件
    """
    # 先将图片保存为PNG
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    png_data = buffer.getvalue()
    
    # PNG文件结构：
    # 8字节签名 + 若干个chunk
    # 每个chunk: 4字节长度 + 4字节类型 + 数据 + 4字节CRC
    
    signature = png_data[:8]
    chunks = []
    pos = 8
    
    while pos < len(png_data):
        length = struct.unpack('>I', png_data[pos:pos+4])[0]
        chunk_type_bytes = png_data[pos+4:pos+8]
        data = png_data[pos+8:pos+8+length]
        crc = png_data[pos+8+length:pos+12+length]
        
        chunks.append((chunk_type_bytes, data, crc))
        pos += 12 + length
    
    # 创建自定义块
    custom_chunk_type = chunk_type.encode('ascii')
    custom_crc = zlib.crc32(custom_chunk_type + chunk_data) & 0xffffffff
    custom_chunk = (
        struct.pack('>I', len(chunk_data)) +
        custom_chunk_type +
        chunk_data +
        struct.pack('>I', custom_crc)
    )
    
    # 重建PNG，在IEND之前插入自定义块
    result = io.BytesIO()
    result.write(signature)
    
    for chunk_type_bytes, data, crc in chunks:
        if chunk_type_bytes == b'IEND':
            # 在IEND之前插入自定义块
            result.write(custom_chunk)
        
        result.write(struct.pack('>I', len(data)))
        result.write(chunk_type_bytes)
        result.write(data)
        result.write(crc)
    
    return result.getvalue()


def read_atlas_metadata_from_png(png_data: bytes) -> Optional[List[Tuple[int, int, int, int, int, int]]]:
    """
    从PNG文件读取地图集元数据
    返回列表：[(x, y, w, h, nx, ny), ...]
    """
    pos = 8  # 跳过PNG签名
    
    while pos < len(png_data):
        length = struct.unpack('>I', png_data[pos:pos+4])[0]
        chunk_type = png_data[pos+4:pos+8].decode('ascii', errors='ignore')
        data = png_data[pos+8:pos+8+length]
        
        if chunk_type == 'aTLS':
            # 明文格式
            text = data.decode('utf-8')
            values = [int(v) for v in text.split(',')]
            result = []
            for i in range(0, len(values), 6):
                result.append(tuple(values[i:i+6]))
            return result
        
        elif chunk_type == 'aTLZ':
            # 二进制格式
            result = []
            for i in range(0, len(data), 12):
                values = struct.unpack('>6H', data[i:i+12])
                result.append(values)
            return result
        
        pos += 12 + length
    
    return None


class ImageAtlasNode:
    """
    纹理地图集生成节点
    将多张图片打包成一张纹理地图集
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_width": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "display": "number"
                }),
                "gap": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "display": "number"
                }),
                "algorithm": ([
                    "Shelf (架子算法)",
                    "MaxRects-BSSF (最短边适配)",
                    "MaxRects-BLSF (最长边适配)",
                    "MaxRects-BAF (最佳面积适配)",
                    "MaxRects-BL (左下角优先)",
                    "Skyline (天际线算法)",
                    "Guillotine (切割算法)",
                ],),
                "trim_mode": ([
                    "alpha (仅透明像素)",
                    "white (仅白色像素)",
                    "both (透明+白色)",
                    "auto (自动检测)",
                    "none (不裁剪)",
                ],),
                "white_threshold": ("INT", {
                    "default": 250,
                    "min": 200,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "size_align": ([
                    "mul4 (4的倍数)",
                    "mul2 (2的倍数)",
                    "pow2 (2的幂)",
                    "none (不贴紧)",
                ],),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("atlas_image", "atlas_json",)
    FUNCTION = "create_atlas"
    CATEGORY = "image/atlas"
    
    def create_atlas(self, images: torch.Tensor, max_width: int, gap: int,
                     algorithm: str, trim_mode: str, white_threshold: int, size_align: str):
        """
        创建纹理地图集
        
        Args:
            images: 输入图片批次 [B, H, W, C]
            max_width: 地图集最大宽度
            gap: 图片间隙（像素）
            metadata_format: 元数据格式
            trim_transparent: 是否裁剪透明边缘
        
        Returns:
            atlas_image: 合并后的纹理图
            metadata_json: JSON格式的元数据
        """
        # 转换为numpy数组
        batch_size = images.shape[0]
        images_np = (images.cpu().numpy() * 255).astype(np.uint8)
        
        # 解析裁剪模式
        if "none" in trim_mode:
            actual_trim_mode = None  # 不裁剪
        elif "alpha" in trim_mode:
            actual_trim_mode = "alpha"
        elif "white" in trim_mode:
            actual_trim_mode = "white"
        elif "both" in trim_mode:
            actual_trim_mode = "both"
        else:
            actual_trim_mode = "auto"
        
        # 处理每张图片，找出非透明/非白色区域
        rect_list: List[RectangleArgs] = []
        
        for i in range(batch_size):
            img = images_np[i]
            
            if actual_trim_mode is not None:
                x, y, w, h = find_opaque_bounds(img, actual_trim_mode, white_threshold)
                if w == 0 or h == 0:
                    # 完全空白，使用1x1
                    cropped = np.zeros((1, 1, img.shape[2]), dtype=np.uint8)
                    w, h = 1, 1
                    x, y = 0, 0
                else:
                    cropped = img[y:y+h, x:x+w]
            else:
                # 不裁剪
                x, y = 0, 0
                h, w = img.shape[:2]
                cropped = img
            
            rect = RectangleArgs(
                width=w, height=h,
                orig_x=x, orig_y=y, orig_w=w, orig_h=h,
                image_index=i, image_data=cropped
            )
            rect_list.append(rect)
        
        # 根据选择的算法进行打包
        if "Shelf" in algorithm:
            packer = FloorPlane()
            packed_rects = packer.deal_rectangle_args_array(rect_list, max_width, gap)
        elif "MaxRects" in algorithm:
            packer = MaxRects()
            if "BSSF" in algorithm:
                heuristic = "BSSF"
            elif "BLSF" in algorithm:
                heuristic = "BLSF"
            elif "BAF" in algorithm:
                heuristic = "BAF"
            else:
                heuristic = "BL"
            packed_rects = packer.deal_rectangle_args_array(rect_list, max_width, gap, heuristic)
        elif "Skyline" in algorithm:
            packer = Skyline()
            packed_rects = packer.deal_rectangle_args_array(rect_list, max_width, gap)
        elif "Guillotine" in algorithm:
            packer = Guillotine()
            packed_rects = packer.deal_rectangle_args_array(rect_list, max_width, gap)
        else:
            # 默认使用 Shelf
            packer = FloorPlane()
            packed_rects = packer.deal_rectangle_args_array(rect_list, max_width, gap)
        
        # 计算最终尺寸
        atlas_width = 0
        atlas_height = 0
        for rect in packed_rects:
            atlas_width = max(atlas_width, rect.x + rect.width)
            atlas_height = max(atlas_height, rect.y + rect.height)
        
        # 确保尺寸至少为1
        atlas_width = max(1, atlas_width)
        atlas_height = max(1, atlas_height)
        
        # 尺寸对齐
        def align_to_mul(val, mul):
            """对齐到指定倍数"""
            return ((val + mul - 1) // mul) * mul
        
        def align_to_pow2(val):
            """对齐到2的幂"""
            if val <= 0:
                return 1
            # 找到大于等于val的最小2的幂
            power = 1
            while power < val:
                power *= 2
            return power
        
        if "mul4" in size_align:
            atlas_width = align_to_mul(atlas_width, 4)
            atlas_height = align_to_mul(atlas_height, 4)
        elif "mul2" in size_align:
            atlas_width = align_to_mul(atlas_width, 2)
            atlas_height = align_to_mul(atlas_height, 2)
        elif "pow2" in size_align:
            atlas_width = align_to_pow2(atlas_width)
            atlas_height = align_to_pow2(atlas_height)
        # none: 不做任何对齐
        
        # 创建地图集图片
        atlas = np.zeros((atlas_height, atlas_width, 4), dtype=np.uint8)
        
        # 放置图片
        for rect in packed_rects:
            if rect.image_data is not None:
                img_data = rect.image_data
                h, w = img_data.shape[:2]
                
                # 确保是4通道
                if img_data.shape[2] == 3:
                    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                    img_data = np.concatenate([img_data, alpha], axis=2)
                
                atlas[rect.y:rect.y+h, rect.x:rect.x+w] = img_data
        
        # 构建元数据
        # 按原始索引排序以保持顺序
        sorted_rects = sorted(packed_rects, key=lambda r: r.image_index)
        
        metadata_list = []
        for rect in sorted_rects:
            # x, y, w, h: 原始图片中非透明区域
            # nx, ny: 在合并纹理中的位置
            metadata_list.append({
                "index": rect.image_index,
                "orig_x": rect.orig_x,
                "orig_y": rect.orig_y,
                "orig_w": rect.orig_w,
                "orig_h": rect.orig_h,
                "atlas_x": rect.x,
                "atlas_y": rect.y
            })
        
        # 生成JSON字符串
        import json
        metadata_json = json.dumps(metadata_list, indent=2)
        
        # 转换回torch tensor用于ComfyUI
        atlas_tensor = torch.from_numpy(atlas.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (atlas_tensor, metadata_json,)


class ImageAtlasSaveNode:
    """
    保存纹理地图集节点
    将地图集保存为带有自定义块的PNG文件
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "atlas_image": ("IMAGE",),
                "atlas_json": ("STRING", {"multiline": True, "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "atlas"}),
                "metadata_format": (["aTLS (明文)", "aTLZ (二进制)"],),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_atlas"
    OUTPUT_NODE = True
    CATEGORY = "image/atlas"
    
    def save_atlas(self, atlas_image: torch.Tensor, atlas_json: str,
                   filename_prefix: str, metadata_format: str):
        """
        保存纹理地图集
        
        Args:
            atlas_image: 地图集图片
            atlas_json: 元数据JSON字符串
            filename_prefix: 文件名前缀
            metadata_format: 元数据格式 (aTLS/aTLZ)
        
        Returns:
            filename: 保存的文件名
        """
        import os
        import json
        import folder_paths
        
        output_dir = folder_paths.get_output_directory()
        
        # 解析元数据
        metadata_list = json.loads(atlas_json) if atlas_json else []
        
        # 获取图片
        img = atlas_image[0]
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        
        # 确保是4通道
        if img_np.shape[2] == 3:
            alpha = np.full((img_np.shape[0], img_np.shape[1], 1), 255, dtype=np.uint8)
            img_np = np.concatenate([img_np, alpha], axis=2)
        
        img_pil = Image.fromarray(img_np, 'RGBA')
        
        # 生成文件名
        counter = 0
        while True:
            filename = f"{filename_prefix}_{counter:05d}.png"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        # 创建块数据
        if metadata_list:
            if "aTLS" in metadata_format:
                values = []
                for m in metadata_list:
                    values.extend([
                        m['orig_x'], m['orig_y'], m['orig_w'], m['orig_h'],
                        m['atlas_x'], m['atlas_y']
                    ])
                chunk_data = ','.join(map(str, values)).encode('utf-8')
                chunk_type = 'aTLS'
            else:
                chunk_data = b''
                for m in metadata_list:
                    chunk_data += struct.pack('>6H',
                        min(65535, m['orig_x']),
                        min(65535, m['orig_y']),
                        min(65535, m['orig_w']),
                        min(65535, m['orig_h']),
                        min(65535, m['atlas_x']),
                        min(65535, m['atlas_y'])
                    )
                chunk_type = 'aTLZ'
            
            # 创建带自定义块的PNG
            png_bytes = create_png_with_custom_chunk(img_pil, chunk_type, chunk_data)
            
            with open(filepath, 'wb') as f:
                f.write(png_bytes)
        else:
            # 没有元数据，普通保存
            img_pil.save(filepath, 'PNG')
        
        return (filename,)


class ImageAtlasExtractNode:
    """
    从地图集提取单张图片节点
    根据索引从地图集中提取指定图片
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "atlas_image": ("IMAGE",),
                "metadata_json": ("STRING", {"multiline": True}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("extracted_image",)
    FUNCTION = "extract"
    CATEGORY = "image/atlas"
    
    def extract(self, atlas_image: torch.Tensor, metadata_json: str, index: int):
        """从地图集提取单张图片"""
        import json
        
        metadata_list = json.loads(metadata_json)
        
        if index >= len(metadata_list):
            raise ValueError(f"索引 {index} 超出范围，地图集只有 {len(metadata_list)} 张图片")
        
        meta = metadata_list[index]
        
        # 获取地图集图片
        atlas_np = (atlas_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # 提取区域
        x, y = meta['atlas_x'], meta['atlas_y']
        w, h = meta['orig_w'], meta['orig_h']
        
        extracted = atlas_np[y:y+h, x:x+w]
        
        # 转换为tensor
        result = torch.from_numpy(extracted.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result,)


class ImageAtlasLoadNode:
    """
    加载纹理地图集节点
    从PNG文件读取地图集及其元数据
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        
        # 获取输入目录中的所有PNG文件
        import os
        files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith('.png'):
                    files.append(f)
        files.sort()
        
        return {
            "required": {
                "image": (files, {"image_upload": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("atlas_image", "atlas_json",)
    FUNCTION = "load_atlas"
    CATEGORY = "image/atlas"
    
    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        import folder_paths
        image_path = folder_paths.get_annotated_filepath(image)
        import os
        return os.path.getmtime(image_path)
    
    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        import folder_paths
        if not folder_paths.exists_annotated_filepath(image):
            return f"无效的图片路径: {image}"
        return True
    
    def load_atlas(self, image: str):
        """加载纹理地图集"""
        import json
        import folder_paths
        
        image_path = folder_paths.get_annotated_filepath(image)
        
        with open(image_path, 'rb') as f:
            png_data = f.read()
        
        # 读取元数据
        metadata = read_atlas_metadata_from_png(png_data)
        
        if metadata:
            metadata_list = []
            for i, (ox, oy, ow, oh, ax, ay) in enumerate(metadata):
                metadata_list.append({
                    "index": i,
                    "orig_x": ox,
                    "orig_y": oy,
                    "orig_w": ow,
                    "orig_h": oh,
                    "atlas_x": ax,
                    "atlas_y": ay
                })
            metadata_json = json.dumps(metadata_list, indent=2)
        else:
            metadata_json = "[]"
        
        # 加载图片
        img = Image.open(io.BytesIO(png_data))
        img_np = np.array(img.convert('RGBA'))
        
        # 转换为tensor
        result = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result, metadata_json,)


class ImageAtlasLoaderNode:
    """
    地图集还原节点
    从带有aTLS/aTLZ块的PNG中提取所有原始图片
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        
        # 获取输入目录中的所有PNG文件
        import os
        files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith('.png'):
                    files.append(f)
        files.sort()
        
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "original_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "original_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("images", "count",)
    FUNCTION = "load_and_extract"
    CATEGORY = "image/atlas"
    
    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        import folder_paths
        image_path = folder_paths.get_annotated_filepath(image)
        import os
        return os.path.getmtime(image_path)
    
    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        import folder_paths
        if not folder_paths.exists_annotated_filepath(image):
            return f"无效的图片路径: {image}"
        return True
    
    def load_and_extract(self, image: str, original_width: int, original_height: int):
        """
        从地图集PNG中提取所有原始图片
        
        Args:
            image: 选择的图片文件名
            original_width: 原始图片宽度
            original_height: 原始图片高度
        
        Returns:
            images: 还原后的图片批次
            count: 图片数量
        """
        import folder_paths
        
        # 获取完整路径
        image_path = folder_paths.get_annotated_filepath(image)
        
        with open(image_path, 'rb') as f:
            png_data = f.read()
        
        # 读取元数据
        metadata = read_atlas_metadata_from_png(png_data)
        
        if metadata is None:
            raise ValueError(f"PNG文件中未找到 aTLS 或 aTLZ 块，无法提取图片: {image_path}")
        
        if len(metadata) == 0:
            raise ValueError(f"PNG文件中的元数据为空: {image_path}")
        
        # 加载地图集图片
        img = Image.open(io.BytesIO(png_data))
        atlas_np = np.array(img.convert('RGBA'))
        
        # 提取每张图片并还原到原始尺寸
        extracted_images = []
        
        for i, (orig_x, orig_y, orig_w, orig_h, atlas_x, atlas_y) in enumerate(metadata):
            # 创建原始尺寸的透明图片
            restored = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            
            # 从地图集中提取区域
            cropped = atlas_np[atlas_y:atlas_y+orig_h, atlas_x:atlas_x+orig_w]
            
            # 计算放置位置（确保不超出边界）
            paste_x = min(orig_x, original_width - 1)
            paste_y = min(orig_y, original_height - 1)
            paste_w = min(orig_w, original_width - paste_x)
            paste_h = min(orig_h, original_height - paste_y)
            
            # 放置到原始位置
            if paste_w > 0 and paste_h > 0:
                restored[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = cropped[:paste_h, :paste_w]
            
            extracted_images.append(restored)
        
        # 堆叠为批次
        batch = np.stack(extracted_images, axis=0)
        
        # 转换为tensor
        result = torch.from_numpy(batch.astype(np.float32) / 255.0)
        
        return (result, len(metadata),)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageAtlas": ImageAtlasNode,
    "ImageAtlasSave": ImageAtlasSaveNode,
    "ImageAtlasExtract": ImageAtlasExtractNode,
    "ImageAtlasLoad": ImageAtlasLoadNode,
    "ImageAtlasLoader": ImageAtlasLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAtlas": "🗺️ 纹理地图集生成",
    "ImageAtlasSave": "💾 保存纹理地图集",
    "ImageAtlasExtract": "✂️ 从地图集提取图片",
    "ImageAtlasLoad": "📂 加载纹理地图集",
    "ImageAtlasLoader": "📦 地图集还原为多图",
}

