using Javis, JavisNB, Animations, LinearAlgebra

dir = "animations/"

function ground(args...)
    background("white")
    sethue("black")
end

function object(p=O, color="black", size=25)
    sethue(color)
    circle(p, size, :fill)
    return p
end
    

function draw_spring(A, B, num_peaks)
    width = 10
    gap = 30
    if(A.x > B.x)
        A, B = B, A
    end
    d, θ = norm(B - A), π/2 - atan((B-A)...)
    Δx = (d - 2*gap)/num_peaks
    xs = vcat([0, gap], gap .+ Δx/2 .+ Δx .* (0:(num_peaks-1)), [gap + num_peaks*Δx, 2*gap + num_peaks*Δx])
    ys = width * vcat([0, 0], (-1).^(0:(num_peaks-1)), [0, 0])
    points = Point.(xs, ys)  	
    Javis.translate(A)
    Javis.rotate(θ)
    sethue("black")
    line.(points[1:end-1], points[2:end], :stroke)
    Javis.rotate(-θ)
    Javis.translate(-A)
end

function solution_animation(sol, i, anchor=O, vert=false, eq = 0, scale = 10)
    function coordToPoint(y)
        vert ? Point(anchor.x,y*scale + eq) : Point(y*scale + eq, anchor.y)
    end
    nframes = length(sol.t)
    mpath = coordToPoint.(reshape(reduce(vcat, sol), length(sol[1]), length(sol))[i, :])
    Javis.Animation(
                range(0, stop = 1, length = nframes),
                mpath, #scaled by 10
                [linear() for _ in 1:(nframes-1)])
end

function spring_animation(sol, i, num_peaks, anchor, color, vert=true, eq = 0, scale=10)
    nframes = length(sol)
    m = Object(1:nframes, (args...) -> object(O, color), Point(0, 0))

    act!(m, Action(1:nframes, solution_animation(sol, i, anchor, vert, eq, scale), Javis.translate()))
    l = Object(1:nframes, (args...)->draw_spring(anchor, pos(m), num_peaks))
end

function animate_spring(sol, color, x0, num_peaks, fname; vid_height = 500)
    begin 
        vid = Video(500, vid_height)
        nframes = length(sol)
        Background(1:nframes, ground)

        spring_animation(sol, 2, num_peaks, Point(x0, -vid_height/2), color)
        embed(vid, pathname = dir*fname)
    end
end