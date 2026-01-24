
const { calculateGrid, is_all_same_aspect_ratio } = require("../../web/scripts/gridUtils.js");

describe("grid utils", () => {
    describe("calculateGrid", () => {
        it("should calculate correct grid for square aspect ratio preference (width > height)", () => {
            const w = 200;
            const h = 100;
            const n = 4;
            // logic: w > h -> cellsize=h=100. cols=ceil(200/100)=2. rows=ceil(4/2)=2.
            // cell_size = min(200/2, 100/2) = 50.
            const result = calculateGrid(w, h, n);
            expect(result).toEqual({ cell_size: 50, columns: 2, rows: 2 });
        });

        it("should handle single image", () => {
            const w = 100;
            const h = 100;
            const n = 1;
            // logic: w > h (false) -> cellsize=w=100. rows=ceil(100/100)=1. cols=ceil(1/1)=1.
            // cell_size = 100.
            const result = calculateGrid(w, h, n);
            expect(result).toEqual({ cell_size: 100, columns: 1, rows: 1 });
        });

         it("should handle width < height", () => {
            const w = 100;
            const h = 200;
            const n = 4;
            // logic: w > h (false) -> cellsize=w=100. rows=ceil(200/100)=2. cols=ceil(4/2)=2.
            // cell_size = min(100/2, 200/2) = 50.
            const result = calculateGrid(w, h, n);
             expect(result).toEqual({ cell_size: 50, columns: 2, rows: 2 });
        });

        it("should handle non-fitting initial guess (n large)", () => {
            const w = 100;
            const h = 100;
            const n = 5;
            // logic: w > h (false) -> cellsize=w=100. rows=1. cols=5.
            // cell_size = min(100/5, 100/1) = 20.
            const result = calculateGrid(w, h, n);
            expect(result).toEqual({ cell_size: 20, columns: 5, rows: 1 });
        });
    });

    describe("is_all_same_aspect_ratio", () => {
        it("should return true for same aspect ratios", () => {
            const imgs = [
                { naturalWidth: 100, naturalHeight: 100 },
                { naturalWidth: 50, naturalHeight: 50 },
            ];
            expect(is_all_same_aspect_ratio(imgs)).toBe(true);
        });

        it("should return false for different aspect ratios", () => {
             const imgs = [
                { naturalWidth: 100, naturalHeight: 100 },
                { naturalWidth: 100, naturalHeight: 50 },
            ];
            expect(is_all_same_aspect_ratio(imgs)).toBe(false);
        });
    });
});
