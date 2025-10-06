using Test
using Statistics

struct WindowedAdaptation
    windowsize::Int64
    windowscale::Int64
    warmup::Int64
    closewindow::Int64
    idx::Base.RefValue{Int64}
    closures::Vector{Int64}
    numwindows::Int64
end

function WindowedAdaptation(warmup; windowsize = 50, windowscale = 2)
    closures = []
    closewindow = windowsize
    j = 0
    if warmup > windowsize
        for w in 1:warmup
            if w == closewindow
                push!(closures, w)
                windowsize *= windowscale
                nextclosewindow = closewindow + windowsize
                if closewindow + windowscale * windowsize >= warmup
                    closewindow = warmup
                else
                    closewindow = nextclosewindow
                end
            end
        end
        numwindows = length(closures)
    else
        numwindows = 0
    end
    return WindowedAdaptation(
        windowsize,
        windowscale,
        warmup,
        windowsize,
        Ref(1),
        closures,
        numwindows
    )
end

function window_closed(wa::WindowedAdaptation, m)
    if wa.warmup < wa.windowsize
        return false
    end
    closed = m == wa.closures[wa.idx[]]
    if closed && wa.idx[] < wa.numwindows
        wa.idx[] += 1
    end
    return closed
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "WindowedAdaptation test" begin
        warmup = 15_000
        iterations = 30_000
        wa = WindowedAdaptation(warmup)
        for m in 1:iterations
            if window_closed(wa, m)
                println(m)
            end
        end
    end
end
