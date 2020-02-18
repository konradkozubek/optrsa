from matplotlib.offsetbox import DraggableAnnotation, AnnotationBbox


class ShiftableAnnotation(DraggableAnnotation):
    def on_pick(self, evt):
        if self._check_still_parented() and evt.artist == self.ref_artist:

            self.mouse_x = evt.mouseevent.x
            self.mouse_y = evt.mouseevent.y
            self.got_artist = True

            if self._use_blit:
                self.ref_artist.set_animated(True)
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(
                                    self.ref_artist.figure.bbox)
                self.ref_artist.draw(self.ref_artist.figure._cachedRenderer)
                self.canvas.blit(self.ref_artist.figure.bbox)
                self._c1 = self.canvas.mpl_connect('motion_notify_event',
                                                   self.on_motion_blit)
            # Without blit don't update offset nor draw canvas during motion:
            # else:
            #     self._c1 = self.canvas.mpl_connect('motion_notify_event',
            #                                        self.on_motion)
            self.save_offset()

    def on_release(self, event):
        if self._check_still_parented() and self.got_artist:
            # Update offset and draw canvas after the button is released:
            dx = event.x - self.mouse_x
            dy = event.y - self.mouse_y
            self.update_offset(dx, dy)
            self.canvas.draw()

            self.finalize_offset()
            self.got_artist = False

            if self._use_blit:
                # Disconnect only when blit is used:
                self.canvas.mpl_disconnect(self._c1)

                self.ref_artist.set_animated(False)


class AnnotationBboxWithShifts(AnnotationBbox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._shiftable = None

    def shiftable(self, state=None, use_blit=False):
        """
        Analogy to setting draggable state.
        Set the shiftable state -- if state is

          * None : toggle the current state

          * True : turn shiftable on

          * False : turn shiftable off

        If shiftable is on, you can shift the annotation on the canvas with
        the mouse.  The ShiftableAnnotation helper instance is returned if
        shiftable is on.
        """
        is_shiftable = self._shiftable is not None

        # if state is None we'll toggle
        if state is None:
            state = not is_shiftable

        if state:
            if self._shiftable is None:
                self._shiftable = ShiftableAnnotation(self, use_blit)
        else:
            if self._shiftable is not None:
                self._shiftable.disconnect()
            self._shiftable = None

        return self._shiftable