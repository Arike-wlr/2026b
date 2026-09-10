import math
import sys


TARGET_RADIUS = 1800.0
ANGLE_ERROR_DEG = 1.0
GRID_STEP = 5.0

sys.stdout.reconfigure(encoding="utf-8")


# 每个检测记录为：(检测点x, 检测点y, 示向度角度)
# 示向度角度：从 x 轴正向逆时针，范围 [0, 360)
DETECTIONS = [
    (0.0, 0.0, 45.0),
    (1000.0, 0.0, 116.565),
    (0.0, 1000.0, 333.435),
]


def normalize_angle(angle_deg):
    return angle_deg % 360.0


def angle_difference_deg(angle_a, angle_b):
    diff = (angle_a - angle_b + 180.0) % 360.0 - 180.0
    return abs(diff)


def bearing_deg(from_x, from_y, to_x, to_y):
    return normalize_angle(math.degrees(math.atan2(to_y - from_y, to_x - from_x)))


def is_inside_target_area(x, y):
    return x * x + y * y <= TARGET_RADIUS * TARGET_RADIUS


def is_feasible_point(x, y, detections):
    for sensor_x, sensor_y, measured_angle in detections:
        true_angle = bearing_deg(sensor_x, sensor_y, x, y)
        if angle_difference_deg(true_angle, measured_angle) > ANGLE_ERROR_DEG:
            return False
    return True


def enumerate_feasible_points(detections, grid_step):
    feasible_points = []
    search_min = -TARGET_RADIUS
    search_max = TARGET_RADIUS
    grid_count = int((search_max - search_min) / grid_step) + 1

    for ix in range(grid_count + 1):
        x = search_min + ix * grid_step
        for iy in range(grid_count + 1):
            y = search_min + iy * grid_step
            if is_inside_target_area(x, y) and is_feasible_point(x, y, detections):
                feasible_points.append((x, y))

    return feasible_points


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def brute_force_diameter(points):
    max_distance = 0.0
    endpoint_a = None
    endpoint_b = None

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            current_distance = distance(points[i], points[j])
            if current_distance > max_distance:
                max_distance = current_distance
                endpoint_a = points[i]
                endpoint_b = points[j]

    return max_distance, endpoint_a, endpoint_b


def diameter_circle_covers_points(points, endpoint_a, endpoint_b, diameter):
    center_x = (endpoint_a[0] + endpoint_b[0]) / 2.0
    center_y = (endpoint_a[1] + endpoint_b[1]) / 2.0
    radius = diameter / 2.0

    for point_x, point_y in points:
        if distance((center_x, center_y), (point_x, point_y)) > radius + 1e-9:
            return False, (center_x, center_y), radius

    return True, (center_x, center_y), radius


def main():
    feasible_points = enumerate_feasible_points(DETECTIONS, GRID_STEP)

    if not feasible_points:
        print("没有枚举到可行点。请减小 GRID_STEP，或检查示向度数据是否自洽。")
        return

    diameter, endpoint_a, endpoint_b = brute_force_diameter(feasible_points)
    covers, circle_center, circle_radius = diameter_circle_covers_points(
        feasible_points,
        endpoint_a,
        endpoint_b,
        diameter,
    )

    print(f"网格步长: {GRID_STEP} m")
    print(f"可行点数量: {len(feasible_points)}")
    print(f"定位区域直径约为: {diameter:.3f} m")
    print(f"直径端点 A: ({endpoint_a[0]:.3f}, {endpoint_a[1]:.3f})")
    print(f"直径端点 B: ({endpoint_b[0]:.3f}, {endpoint_b[1]:.3f})")
    print(f"以该直径为直径的圆心: ({circle_center[0]:.3f}, {circle_center[1]:.3f})")
    print(f"以该直径为直径的圆半径: {circle_radius:.3f} m")

    if covers:
        print("枚举结果: 该圆可以覆盖枚举得到的定位区域。")
    else:
        print("枚举结果: 该圆不能覆盖枚举得到的定位区域。")


if __name__ == "__main__":
    main()
