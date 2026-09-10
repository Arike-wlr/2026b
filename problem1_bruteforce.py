import math
import os
import random
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point, Polygon


TARGET_RADIUS = 1800.0
ANGLE_ERROR_DEG = 1.0
RAY_LENGTH = 10000000.0
BUFFER_RESOLUTION = 256
EPSILON = 1e-9

OUTPUT_DIR = "figures"
REPORT_PATH = "problem1_batch_report.md"
VISUALIZATION_PATH = os.path.join(OUTPUT_DIR, "problem1_cannot_cover_case.png")

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 每个检测记录为：(检测点x, 检测点y, 示向度角度)
# 示向度角度：从 x 轴正向逆时针，范围 [0, 360)
DETECTIONS = [
    (0.0, 0.0, 45.0),
    (1000.0, 0.0, 116.565),
    (0.0, 1000.0, 333.435),
]


@dataclass
class LocationResult:
    detections: list
    region: Polygon
    vertices: list
    diameter: float
    endpoint_a: tuple
    endpoint_b: tuple
    min_circle: tuple
    can_cover: bool


class DegenerateRegionError(Exception):
    pass


def angle_to_vector(angle_deg):
    angle_rad = math.radians(angle_deg)
    return math.cos(angle_rad), math.sin(angle_rad)


def normalize_angle(angle_deg):
    return angle_deg % 360.0


def bearing_deg(from_x, from_y, to_x, to_y):
    return normalize_angle(math.degrees(math.atan2(to_y - from_y, to_x - from_x)))


def sector_polygon(sensor_x, sensor_y, bearing, error_deg, ray_length):
    left_x, left_y = angle_to_vector(bearing - error_deg)
    right_x, right_y = angle_to_vector(bearing + error_deg)

    return Polygon([
        (sensor_x, sensor_y),
        (sensor_x + ray_length * left_x, sensor_y + ray_length * left_y),
        (sensor_x + ray_length * right_x, sensor_y + ray_length * right_y),
    ])


def artificial_bounding_polygon(ray_length):
    return Polygon([
        (-ray_length, -ray_length),
        (ray_length, -ray_length),
        (ray_length, ray_length),
        (-ray_length, ray_length),
    ])


def target_area():
    return Point(0.0, 0.0).buffer(TARGET_RADIUS, resolution=BUFFER_RESOLUTION)


def largest_polygon(geometry):
    if geometry.is_empty:
        return None
    if geometry.geom_type in ("LineString", "LinearRing", "MultiLineString", "Point", "MultiPoint"):
        raise DegenerateRegionError("定位区域退化为线或点")
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        polygons = [item for item in geometry.geoms if item.geom_type == "Polygon"]
        degenerate_items = [
            item for item in geometry.geoms
            if item.geom_type in ("LineString", "LinearRing", "MultiLineString", "Point", "MultiPoint")
        ]
        if polygons:
            return max(polygons, key=lambda item: item.area)
        if degenerate_items:
            raise DegenerateRegionError("定位区域退化为线或点")
    return None


def intersection_polygon(detections):
    current_region = artificial_bounding_polygon(RAY_LENGTH)

    for sensor_x, sensor_y, bearing in detections:
        current_region = current_region.intersection(
            sector_polygon(sensor_x, sensor_y, bearing, ANGLE_ERROR_DEG, RAY_LENGTH)
        )
        if current_region.is_empty:
            return None

    current_region = current_region.intersection(target_area())
    return largest_polygon(current_region)


def polygon_vertices(polygon):
    return list(polygon.exterior.coords)[:-1]


def touches_artificial_boundary(vertices):
    boundary_limit = RAY_LENGTH * 0.999
    return any(abs(x) >= boundary_limit or abs(y) >= boundary_limit for x, y in vertices)


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def polygon_diameter(vertices):
    max_distance = 0.0
    endpoint_a = vertices[0]
    endpoint_b = vertices[0]

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            current_distance = distance(vertices[i], vertices[j])
            if current_distance > max_distance:
                max_distance = current_distance
                endpoint_a = vertices[i]
                endpoint_b = vertices[j]

    return max_distance, endpoint_a, endpoint_b


def circle_from_two_points(point_a, point_b):
    center_x = (point_a[0] + point_b[0]) / 2.0
    center_y = (point_a[1] + point_b[1]) / 2.0
    radius = distance(point_a, point_b) / 2.0
    return center_x, center_y, radius


def circle_from_three_points(point_a, point_b, point_c):
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c
    denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if abs(denominator) < 1e-12:
        return None

    center_x = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / denominator
    center_y = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / denominator
    radius = distance((center_x, center_y), point_a)
    return center_x, center_y, radius


def point_outside_circle(point, circle):
    center_x, center_y, radius = circle
    return distance(point, (center_x, center_y)) > radius + EPSILON


def minimum_circle_with_two_boundary_points(points, point_a, point_b):
    circle = circle_from_two_points(point_a, point_b)

    for point in points:
        if point_outside_circle(point, circle):
            candidate = circle_from_three_points(point_a, point_b, point)
            if candidate is not None:
                circle = candidate

    return circle


def minimum_circle_with_one_boundary_point(points, boundary_point):
    circle = (boundary_point[0], boundary_point[1], 0.0)

    for index, point in enumerate(points):
        if point_outside_circle(point, circle):
            if circle[2] == 0.0:
                circle = circle_from_two_points(boundary_point, point)
            else:
                circle = minimum_circle_with_two_boundary_points(
                    points[:index],
                    boundary_point,
                    point,
                )

    return circle


def minimum_enclosing_circle(points):
    shuffled_points = list(points)
    random.seed(2026)
    random.shuffle(shuffled_points)

    circle = (0.0, 0.0, -1.0)
    for index, point in enumerate(shuffled_points):
        if circle[2] < 0.0 or point_outside_circle(point, circle):
            circle = minimum_circle_with_one_boundary_point(shuffled_points[:index], point)
    return circle


def solve_location_region(detections):
    region = intersection_polygon(detections)
    if region is None:
        raise DegenerateRegionError("无交会区域")

    vertices = polygon_vertices(region)
    if len(vertices) < 3:
        raise DegenerateRegionError("定位区域退化为线或点")
    if touches_artificial_boundary(vertices):
        raise DegenerateRegionError("无有限交会区域")

    diameter, endpoint_a, endpoint_b = polygon_diameter(vertices)
    min_circle = minimum_enclosing_circle(vertices)
    can_cover = min_circle[2] <= diameter / 2.0 + EPSILON

    return LocationResult(
        detections=detections,
        region=region,
        vertices=vertices,
        diameter=diameter,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        min_circle=min_circle,
        can_cover=can_cover,
    )


def random_point_in_target(rng, max_radius=1500.0):
    radius = max_radius * math.sqrt(rng.random())
    angle = 2.0 * math.pi * rng.random()
    return radius * math.cos(angle), radius * math.sin(angle)


def generate_random_case(rng, sensor_count):
    source_x, source_y = random_point_in_target(rng, max_radius=1200.0)
    detections = []

    for _ in range(sensor_count):
        while True:
            sensor_x, sensor_y = random_point_in_target(rng, max_radius=1700.0)
            if distance((sensor_x, sensor_y), (source_x, source_y)) > 80.0:
                break

        true_bearing = bearing_deg(sensor_x, sensor_y, source_x, source_y)
        measured_bearing = normalize_angle(true_bearing + rng.uniform(-0.65, 0.65))
        detections.append((sensor_x, sensor_y, measured_bearing))

    return detections


def generate_batch_results():
    rng = random.Random(20260910)
    sensor_counts = [2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
    results = []
    cannot_cover_result = None

    for case_index, sensor_count in enumerate(sensor_counts, start=1):
        for _ in range(300):
            detections = generate_random_case(rng, sensor_count)
            try:
                result = solve_location_region(detections)
                results.append((case_index, result, None))
                if not result.can_cover and cannot_cover_result is None:
                    cannot_cover_result = result
                break
            except DegenerateRegionError as error:
                last_error = str(error)
        else:
            results.append((case_index, None, last_error))

    if cannot_cover_result is None:
        for _ in range(2000):
            detections = generate_random_case(rng, rng.choice([3, 4, 5]))
            try:
                result = solve_location_region(detections)
                if not result.can_cover:
                    cannot_cover_result = result
                    break
            except DegenerateRegionError:
                pass

    return results, cannot_cover_result


def write_batch_report(results, cannot_cover_result):
    lines = [
        "# 问题1解析几何法批量测试报告",
        "",
        "## 方法说明",
        "",
        "每个检测点根据示向度 `theta_i` 和误差范围 `±1°` 生成一个由两条射线夹成的扇形区域。",
        "程序用 Shapely 对所有扇形求交，并将最终交集再次与 `Point(0,0).buffer(1800)` 相交，确保定位区域严格落在目标圆域内。",
        "若交集为空，或仅为线段/点，则记为无有效交会区域；若得到多边形，则枚举所有顶点对求直径 `d`，并用最小外接圆半径 `Rmin` 判断以 `d` 为直径的圆是否覆盖定位区域。",
        "",
        "判据：若 `Rmin > d/2 + 1e-9`，则不能覆盖；否则可覆盖。",
        "",
        "## 批量结果",
        "",
        "| 编号 | N | 顶点数 | 面积(m^2) | d(m) | d/2(m) | Rmin(m) | 结论 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for case_index, result, error in results:
        if result is None:
            lines.append(f"| {case_index} | - | - | - | - | - | - | {error or '无交会区域'} |")
            continue
        conclusion = "可以覆盖" if result.can_cover else "不能覆盖"
        lines.append(
            f"| {case_index} | {len(result.detections)} | {len(result.vertices)} | "
            f"{result.region.area:.6f} | {result.diameter:.6f} | "
            f"{result.diameter / 2.0:.6f} | {result.min_circle[2]:.6f} | {conclusion} |"
        )

    lines += [
        "",
        "## 不能覆盖样例",
        "",
    ]

    if cannot_cover_result is None:
        lines.append("本次随机样例中未找到不能覆盖案例。")
    else:
        lines.append(f"可视化文件：`{VISUALIZATION_PATH}`")
        lines.append("")
        lines.append("| 检测点 | x | y | 示向度(度) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for index, (sensor_x, sensor_y, bearing) in enumerate(cannot_cover_result.detections, start=1):
            lines.append(f"| S{index} | {sensor_x:.6f} | {sensor_y:.6f} | {bearing:.6f} |")
        lines.append("")
        lines.append(f"- 顶点数：{len(cannot_cover_result.vertices)}")
        lines.append(f"- 直径 d：{cannot_cover_result.diameter:.9f} m")
        lines.append(f"- d/2：{cannot_cover_result.diameter / 2.0:.9f} m")
        lines.append(f"- Rmin：{cannot_cover_result.min_circle[2]:.9f} m")
        lines.append("- 结论：以 d 为直径的圆不能覆盖该定位区域")

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def plot_cannot_cover_case(result):
    if result is None:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    vx, vy = result.region.exterior.xy
    ax.add_patch(MplPolygon(
        list(zip(vx, vy)),
        closed=True,
        facecolor="#8b0000",
        alpha=0.5,
        edgecolor="black",
        linewidth=2.6,
        label="定位区域",
        zorder=5,
    ))

    ax.plot(
        [result.endpoint_a[0], result.endpoint_b[0]],
        [result.endpoint_a[1], result.endpoint_b[1]],
        color="black",
        linewidth=2.4,
        label="直径 d",
        zorder=8,
    )

    diameter_circle = circle_from_two_points(result.endpoint_a, result.endpoint_b)
    ax.add_patch(MplCircle(
        (diameter_circle[0], diameter_circle[1]),
        diameter_circle[2],
        fill=False,
        linestyle="--",
        edgecolor="darkorange",
        linewidth=1.0,
        label="直径圆",
        zorder=7,
    ))
    ax.add_patch(MplCircle(
        (result.min_circle[0], result.min_circle[1]),
        result.min_circle[2],
        fill=False,
        linestyle="-.",
        edgecolor="green",
        linewidth=1.0,
        label="最小外接圆",
        zorder=9,
    ))

    center_x = result.region.centroid.x
    center_y = result.region.centroid.y
    ax.set_xlim(center_x - 60.0, center_x + 60.0)
    ax.set_ylim(center_y - 60.0, center_y + 60.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("图1：Rmin > d/2 的不可覆盖情况（局部放大）", fontsize=14, pad=12)

    region_corner = max(result.vertices, key=lambda point: distance(point, (result.min_circle[0], result.min_circle[1])))
    min_circle_label_point = (
        result.min_circle[0] + result.min_circle[2] / math.sqrt(2.0),
        result.min_circle[1] + result.min_circle[2] / math.sqrt(2.0),
    )
    diameter_circle_label_point = (
        diameter_circle[0] - diameter_circle[2] / math.sqrt(2.0),
        diameter_circle[1] - diameter_circle[2] / math.sqrt(2.0),
    )

    ax.annotate(
        f"最小外接圆半径 Rmin = {result.min_circle[2]:.3f} m",
        xy=min_circle_label_point,
        xytext=(center_x - 54.0, center_y + 46.0),
        arrowprops={"arrowstyle": "->", "color": "green", "linewidth": 1.8},
        color="green",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "green", "alpha": 0.86},
        zorder=10,
    )
    ax.annotate(
        f"直径圆半径 d/2 = {diameter_circle[2]:.3f} m",
        xy=diameter_circle_label_point,
        xytext=(center_x - 54.0, center_y - 52.0),
        arrowprops={"arrowstyle": "->", "color": "darkorange", "linewidth": 1.8},
        color="darkorange",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "darkorange", "alpha": 0.86},
        zorder=10,
    )
    ax.annotate(
        "超出直径圆的区域角点",
        xy=region_corner,
        xytext=(center_x + 9.0, center_y + 38.0),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.4},
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "black", "alpha": 0.82},
        zorder=10,
    )

    ax.legend(loc="lower right", fontsize=8, framealpha=0.82)
    fig.savefig(VISUALIZATION_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_single_result():
    try:
        result = solve_location_region(DETECTIONS)
    except DegenerateRegionError as error:
        print(str(error))
        return

    print(f"检测点数量: {len(result.detections)}")
    print(f"定位区域顶点数量: {len(result.vertices)}")
    print("定位区域顶点:")
    for index, (vertex_x, vertex_y) in enumerate(result.vertices, start=1):
        print(f"  P{index}: ({vertex_x:.9f}, {vertex_y:.9f})")
    print(f"定位区域面积: {result.region.area:.9f}")
    print(f"定位区域直径 d: {result.diameter:.9f}")
    print(f"直径端点 A: ({result.endpoint_a[0]:.9f}, {result.endpoint_a[1]:.9f})")
    print(f"直径端点 B: ({result.endpoint_b[0]:.9f}, {result.endpoint_b[1]:.9f})")
    print(f"最小外接圆圆心: ({result.min_circle[0]:.9f}, {result.min_circle[1]:.9f})")
    print(f"最小外接圆半径 Rmin: {result.min_circle[2]:.9f}")
    print(f"d / 2: {result.diameter / 2.0:.9f}")
    print("结论: 以 d 为直径的圆" + ("可以" if result.can_cover else "不能") + "覆盖该定位区域")


def main():
    print_single_result()

    results, cannot_cover_result = generate_batch_results()
    write_batch_report(results, cannot_cover_result)
    plot_cannot_cover_case(cannot_cover_result)

    print(f"\n批量测试报告已生成: {REPORT_PATH}")
    if cannot_cover_result is not None:
        print(f"不能覆盖样例图已生成: {VISUALIZATION_PATH}")
    else:
        print("本次随机测试未找到不能覆盖样例。")


if __name__ == "__main__":
    main()
